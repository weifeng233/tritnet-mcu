/*
 * motor_control_mlp.h — Motor control application layer
 *
 * MLP-based motor controller that replaces or augments a PID system.
 * Two modes:
 *   1. Direct MLP control: MLP outputs control signal directly
 *   2. Gain scheduling: MLP predicts Kp, Ki, Kd for a PID controller
 *
 * Input features encode PID-like information, enabling the MLP to
 * learn non-linear corrections over traditional linear PID.
 *
 * Hybrid control architecture (recommended):
 *   Outer loop (100-500Hz): MLP position controller
 *   Inner loop (1-5kHz):    PID velocity controller
 *   Innermost (20-50kHz):   PI current controller
 */

#ifndef MOTOR_CONTROL_MLP_H
#define MOTOR_CONTROL_MLP_H

#include "mlp_core.h"

/* ================================================================== */
/* Control mode selection                                              */
/* ================================================================== */

typedef enum {
  MC_MODE_DIRECT    = 0,  /* MLP outputs control signal directly */
  MC_MODE_GAIN_SCHED = 1  /* MLP predicts PID gains (Kp, Ki, Kd) */
} mc_mode_t;

/* ================================================================== */
/* Motor control state (persistent across control loop iterations)     */
/* ================================================================== */

typedef struct {
  /* Error accumulators */
  float e_pos_prev;      /* previous position error (for derivative) */
  float e_pos_integral;  /* integral of position error (for I-term)  */
  float e_pos_prev2;     /* error two steps ago (for delta features) */

  /* Control history */
  float u_prev;          /* previous control output */
  float u_prev2;         /* two-step-ago control output */

  /* Physical limits (configured at init) */
  float pos_limit_min;   /* min position (physical units) */
  float pos_limit_max;   /* max position */
  float vel_limit_max;   /* max velocity */
  float ctrl_limit_min;  /* min control output (e.g., -100% PWM) */
  float ctrl_limit_max;  /* max control output (e.g., +100% PWM) */
  float integral_limit;  /* anti-windup clamp on integral error */

  /* Normalization scales (based on system physical limits) */
  float norm_pos;        /* position normalization factor (1/range) */
  float norm_vel;        /* velocity normalization factor */
  float norm_ctrl;       /* control output normalization factor */

  /* Mode */
  mc_mode_t mode;

  /* Gain scheduling mode parameters */
  float kp_min, kp_max;  /* valid Kp range */
  float ki_min, ki_max;  /* valid Ki range */
  float kd_min, kd_max;  /* valid Kd range */
  float dt;              /* control loop period (seconds) */

} mc_state_t;

/* ================================================================== */
/* Initialization                                                      */
/* ================================================================== */

static inline void mc_init(mc_state_t *s, mc_mode_t mode, float dt) {
  s->e_pos_prev     = 0.0f;
  s->e_pos_integral = 0.0f;
  s->e_pos_prev2    = 0.0f;
  s->u_prev         = 0.0f;
  s->u_prev2        = 0.0f;
  s->mode           = mode;
  s->dt             = dt;

  /* Default physical limits (should be configured per-system) */
  s->pos_limit_min  = -180.0f;
  s->pos_limit_max  =  180.0f;
  s->vel_limit_max  =  1000.0f;
  s->ctrl_limit_min = -100.0f;
  s->ctrl_limit_max =  100.0f;
  s->integral_limit =  50.0f;

  /* Default normalization: map physical range to [-1, +1] */
  float pos_range = s->pos_limit_max - s->pos_limit_min;
  s->norm_pos  = (pos_range > 0.0f) ? (2.0f / pos_range) : 1.0f;
  s->norm_vel  = (s->vel_limit_max > 0.0f) ? (1.0f / s->vel_limit_max) : 1.0f;
  s->norm_ctrl = (s->ctrl_limit_max > 0.0f) ? (1.0f / s->ctrl_limit_max) : 1.0f;

  /* Default gain scheduling ranges */
  s->kp_min = 0.0f;  s->kp_max = 10.0f;
  s->ki_min = 0.0f;  s->ki_max = 5.0f;
  s->kd_min = 0.0f;  s->kd_max = 2.0f;
}

/* Configure normalization from system physical limits */
static inline void mc_set_limits(mc_state_t *s,
                                  float pos_min, float pos_max,
                                  float vel_max,
                                  float ctrl_min, float ctrl_max) {
  s->pos_limit_min  = pos_min;
  s->pos_limit_max  = pos_max;
  s->vel_limit_max  = vel_max;
  s->ctrl_limit_min = ctrl_min;
  s->ctrl_limit_max = ctrl_max;

  float pos_range = pos_max - pos_min;
  s->norm_pos  = (pos_range > 0.0f) ? (2.0f / pos_range) : 1.0f;
  s->norm_vel  = (vel_max > 0.0f) ? (1.0f / vel_max) : 1.0f;
  s->norm_ctrl = (ctrl_max > 0.0f) ? (1.0f / ctrl_max) : 1.0f;
}

/* Set PID gain ranges for gain scheduling mode */
static inline void mc_set_gain_ranges(mc_state_t *s,
                                       float kp_min, float kp_max,
                                       float ki_min, float ki_max,
                                       float kd_min, float kd_max) {
  s->kp_min = kp_min; s->kp_max = kp_max;
  s->ki_min = ki_min; s->ki_max = ki_max;
  s->kd_min = kd_min; s->kd_max = kd_max;
}

/* ================================================================== */
/* Input feature construction — PID-aware features                     */
/* ================================================================== */

/*
 * mc_build_features: construct normalized input vector for MLP
 *
 * Input features (12 dimensions):
 *   [0]  e_pos(t)          = setpoint - actual           (position error)
 *   [1]  e_vel(t)          = setpoint_vel - actual_vel   (velocity error)
 *   [2]  delta_e_pos       = e_pos(t) - e_pos(t-1)      (error change)
 *   [3]  e_pos_integral    = ∫ e_pos dt                  (integral, anti-windup clamped)
 *   [4]  actual_pos_norm   = (actual - center) * norm    (normalized position)
 *   [5]  actual_vel_norm   = vel * norm                  (normalized velocity)
 *   [6]  current_norm      = current / max_current       (normalized motor current)
 *   [7]  setpoint_pos_norm = (setpoint - center) * norm  (reference position)
 *   [8]  setpoint_vel_norm = setpoint_vel * norm         (reference velocity)
 *   [9]  lookahead         = setpoint(t+1) - setpoint(t) (trajectory lookahead)
 *   [10] u_prev_norm       = u(t-1) * norm_ctrl          (previous control)
 *   [11] u_prev2_norm      = u(t-2) * norm_ctrl          (two-step history)
 */
#define MC_FEATURE_DIM 12

static inline void mc_build_features(mc_state_t *s,
                                      float setpoint_pos, float setpoint_vel,
                                      float actual_pos,   float actual_vel,
                                      float actual_current,
                                      float setpoint_next,  /* lookahead: setpoint at t+1 */
                                      float *features) {
  /* Position error (centered) */
  float pos_center = (s->pos_limit_min + s->pos_limit_max) * 0.5f;
  float e_pos = setpoint_pos - actual_pos;
  float e_vel = setpoint_vel - actual_vel;

  /* Update integral with anti-windup */
  s->e_pos_integral += e_pos * s->dt;
  if (s->e_pos_integral >  s->integral_limit) s->e_pos_integral =  s->integral_limit;
  if (s->e_pos_integral < -s->integral_limit) s->e_pos_integral = -s->integral_limit;

  /* Error change */
  float delta_e = e_pos - s->e_pos_prev;

  /* Normalize all features to approximately [-1, +1] */
  features[0]  = e_pos * s->norm_pos;
  features[1]  = e_vel * s->norm_vel;
  features[2]  = delta_e * s->norm_pos;
  features[3]  = s->e_pos_integral * s->norm_pos;
  features[4]  = (actual_pos - pos_center) * s->norm_pos;
  features[5]  = actual_vel * s->norm_vel;
  features[6]  = actual_current * s->norm_ctrl;
  features[7]  = (setpoint_pos - pos_center) * s->norm_pos;
  features[8]  = setpoint_vel * s->norm_vel;
  features[9]  = (setpoint_next - setpoint_pos) * s->norm_pos;
  features[10] = s->u_prev * s->norm_ctrl;
  features[11] = s->u_prev2 * s->norm_ctrl;
}

/* ================================================================== */
/* Control output: direct MLP mode                                     */
/* ================================================================== */

/*
 * mc_control_direct: MLP outputs control signal directly
 *
 * MLP output (1 dim, tanh bounded) maps to control range:
 *   u_raw = tanh[-1, +1]
 *   u_physical = u_raw * ctrl_limit_max
 *
 * Safety: clamps to physical limits.
 */
static inline float mc_control_direct(mc_state_t *s,
                                       float setpoint_pos, float setpoint_vel,
                                       float actual_pos,   float actual_vel,
                                       float actual_current,
                                       float setpoint_next) {
  float features[MC_FEATURE_DIM];
  mc_build_features(s, setpoint_pos, setpoint_vel,
                     actual_pos, actual_vel, actual_current,
                     setpoint_next, features);

  float mlp_out[1];
  mlp_forward_f32(features, mlp_out);

  /* Scale from [-1, +1] to physical units */
  float u = mlp_out[0] * s->ctrl_limit_max;

  /* Clamp to physical limits */
  if (u > s->ctrl_limit_max) u = s->ctrl_limit_max;
  if (u < s->ctrl_limit_min) u = s->ctrl_limit_min;

  /* Update history */
  s->u_prev2     = s->u_prev;
  s->u_prev      = u;
  s->e_pos_prev2 = s->e_pos_prev;
  s->e_pos_prev  = setpoint_pos - actual_pos;

  return u;
}

/* ================================================================== */
/* Control output: gain scheduling mode                                */
/* ================================================================== */

/*
 * mc_control_gain_sched: MLP predicts Kp, Ki, Kd, then applies PID
 *
 * MLP output (3 dims, tanh bounded):
 *   [0] -> Kp mapped from [kp_min, kp_max]
 *   [1] -> Ki mapped from [ki_min, ki_max]
 *   [2] -> Kd mapped from [kd_min, kd_max]
 *
 * Then: u = Kp*e + Ki*∫e + Kd*(de/dt)
 */
static inline float mc_control_gain_sched(mc_state_t *s,
                                           float setpoint_pos, float setpoint_vel,
                                           float actual_pos,   float actual_vel,
                                           float actual_current,
                                           float setpoint_next) {
  float features[MC_FEATURE_DIM];
  mc_build_features(s, setpoint_pos, setpoint_vel,
                     actual_pos, actual_vel, actual_current,
                     setpoint_next, features);

  float mlp_out[3];
  mlp_forward_f32(features, mlp_out);

  /* Map tanh[-1, +1] to gain ranges */
  float Kp = s->kp_min + (mlp_out[0] + 1.0f) * 0.5f * (s->kp_max - s->kp_min);
  float Ki = s->ki_min + (mlp_out[1] + 1.0f) * 0.5f * (s->ki_max - s->ki_min);
  float Kd = s->kd_min + (mlp_out[2] + 1.0f) * 0.5f * (s->kd_max - s->kd_min);

  /* Standard PID computation */
  float e_pos = setpoint_pos - actual_pos;
  float e_vel = setpoint_vel - actual_vel;
  float de_dt = (e_pos - s->e_pos_prev) / s->dt;

  s->e_pos_integral += e_pos * s->dt;
  if (s->e_pos_integral >  s->integral_limit) s->e_pos_integral =  s->integral_limit;
  if (s->e_pos_integral < -s->integral_limit) s->e_pos_integral = -s->integral_limit;

  float u = Kp * e_pos + Ki * s->e_pos_integral + Kd * de_dt;

  /* Clamp */
  if (u > s->ctrl_limit_max) u = s->ctrl_limit_max;
  if (u < s->ctrl_limit_min) u = s->ctrl_limit_min;

  /* Update history */
  s->u_prev2     = s->u_prev;
  s->u_prev      = u;
  s->e_pos_prev2 = s->e_pos_prev;
  s->e_pos_prev  = e_pos;

  return u;
}

/* ================================================================== */
/* Unified control step                                                */
/* ================================================================== */

static inline float mc_control_step(mc_state_t *s,
                                     float setpoint_pos, float setpoint_vel,
                                     float actual_pos,   float actual_vel,
                                     float actual_current,
                                     float setpoint_next) {
  if (s->mode == MC_MODE_GAIN_SCHED) {
    return mc_control_gain_sched(s, setpoint_pos, setpoint_vel,
                                  actual_pos, actual_vel, actual_current,
                                  setpoint_next);
  } else {
    return mc_control_direct(s, setpoint_pos, setpoint_vel,
                              actual_pos, actual_vel, actual_current,
                              setpoint_next);
  }
}

/* ================================================================== */
/* Reset state (call on system stop/re-init)                           */
/* ================================================================== */

static inline void mc_reset(mc_state_t *s) {
  s->e_pos_prev     = 0.0f;
  s->e_pos_integral = 0.0f;
  s->e_pos_prev2    = 0.0f;
  s->u_prev         = 0.0f;
  s->u_prev2        = 0.0f;
}

#endif /* MOTOR_CONTROL_MLP_H */
