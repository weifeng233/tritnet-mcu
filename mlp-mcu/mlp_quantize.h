/*
 * mlp_quantize.h — INT8 per-channel symmetric quantization
 *
 * Supports:
 *   - Weight quantization: float32 -> int8 (per-output-channel scales)
 *   - Hybrid inference: int8 weights + float32 activations
 *   - Full int8 inference (optional, with activation requantization)
 *
 * Quantization formula:
 *   w_quant[i][j] = clamp(round(w_float[i][j] / scale[i]), -128, 127)
 *   scale[i] = max(|w_float[i][j]|) / 127
 *   output[i] = bias[i] + scale[i] * SUM(w_quant[i][j] * x[j])
 *
 * Memory savings: 75% vs float32 (int8 weight + float32 scale)
 */

#ifndef MLP_QUANTIZE_H
#define MLP_QUANTIZE_H

#include <stdint.h>
#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Quantized weight structure                                          */
/* ------------------------------------------------------------------ */

typedef struct {
  const int8_t  *weights;   /* [n_out][n_in] — stored in Flash */
  const float   *bias;      /* [n_out] — float32 bias in Flash  */
  const float   *scales;    /* [n_out] — per-channel scale      */
  int            n_in;
  int            n_out;
  float          input_scale;   /* activation quant scale (1.0 if not quantized) */
} mlp_quant_layer_t;

/* ------------------------------------------------------------------ */
/* Weight quantization (runs on PC at export time, not on MCU)         */
/* ------------------------------------------------------------------ */

/* Quantize a single output channel's weights to int8
 * Returns the scale factor used for this channel.
 * Input:  w_float[n_in]   — float32 weights for one neuron
 * Output: w_quant[n_in]   — quantized int8 weights (in-place or separate buffer)
 * Returns: scale factor (max_abs / 127.0)
 */
static inline float mlp_quantize_channel(const float *w_float, int n_in,
                                          int8_t *w_quant) {
  float max_abs = 0.0f;
  for (int j = 0; j < n_in; j++) {
    float a = fabsf(w_float[j]);
    if (a > max_abs) max_abs = a;
  }
  float scale = max_abs / 127.0f;
  float inv_scale = (scale > 1e-12f) ? (1.0f / scale) : 0.0f;
  for (int j = 0; j < n_in; j++) {
    int32_t q = (int32_t)roundf(w_float[j] * inv_scale);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    w_quant[j] = (int8_t)q;
  }
  return scale;
}

/* Quantize an entire weight matrix to int8
 * Input:  w_float[n_out][n_in]
 * Output: w_quant[n_out][n_in], scales[n_out]
 */
static inline void mlp_quantize_matrix(const float *w_float, int n_out, int n_in,
                                        int8_t *w_quant, float *scales) {
  for (int r = 0; r < n_out; r++) {
    scales[r] = mlp_quantize_channel(w_float + r * n_in, n_in, w_quant + r * n_in);
  }
}

/* Compute max quantization error for a weight matrix (for quality check) */
static inline float mlp_quantize_max_error(const float *w_float, const int8_t *w_quant,
                                            const float *scales, int n_out, int n_in) {
  float max_err = 0.0f;
  for (int r = 0; r < n_out; r++) {
    float sc = scales[r];
    for (int j = 0; j < n_in; j++) {
      float recon = (float)w_quant[r * n_in + j] * sc;
      float err = fabsf(w_float[r * n_in + j] - recon);
      if (err > max_err) max_err = err;
    }
  }
  return max_err;
}

/* ------------------------------------------------------------------ */
/* Hybrid int8×float32 dot product (int8 weights, float32 activation)  */
/* ------------------------------------------------------------------ */

/* int8 weight × float32 input → float32 output
 * Accumulation in int32 after converting input to q7
 * Returns: scale * SUM(w_q7[j] * round(x_f32[j]/input_scale)) + bias
 *
 * Alternative (more precise but slower): accumulate in float after dequantizing each w
 */

/* Method A: dequantize weight then float dot (simpler, more accurate) */
static inline float mlp_dot_int8_dequant(const int8_t *w_q8, const float *x_f32,
                                          float scale, int n) {
  float acc = 0.0f;
  for (int j = 0; j < n; j++)
    acc += (float)w_q8[j] * x_f32[j];
  return acc * scale;
}

/* Method B: fully integer MAC (faster on Cortex-M0/M3 without FPU)
 * Requires input quantized to int8 first.
 * acc = SUM(w_q8[j] * x_q8[j]) in int32 (safe for n <= 256)
 * output = bias + scale * inv_x_scale * acc
 */
static inline int32_t mlp_dot_int8_raw(const int8_t *w_q8, const int8_t *x_q8, int n) {
  int32_t acc = 0;
  for (int j = 0; j < n; j++)
    acc += (int32_t)w_q8[j] * (int32_t)x_q8[j];
  return acc;
}

/* ------------------------------------------------------------------ */
/* Hybrid linear forward: int8 weights, float32 input                  */
/* ------------------------------------------------------------------ */

static inline void mlp_linear_fwd_int8(const float *x,
                                        const int8_t *w_q8, const float *bias,
                                        const float *scales,
                                        int n_out, int n_in,
                                        float *out) {
  for (int r = 0; r < n_out; r++) {
    float acc = mlp_dot_int8_dequant(w_q8 + r * n_in, x, scales[r], n_in);
    out[r] = acc + (bias ? bias[r] : 0.0f);
  }
}

/* ------------------------------------------------------------------ */
/* Activation quantization (for full-integer inference)                */
/* Used only on Cortex-M0/M3 without FPU                               */
/* ------------------------------------------------------------------ */

/* Determine activation scale via running min/max tracking */
typedef struct {
  float min_val;
  float max_val;
  float scale;   /* max(|min|, |max|) / 127 */
} mlp_act_quant_calibrator_t;

static inline void mlp_calibrator_reset(mlp_act_quant_calibrator_t *cal) {
  cal->min_val =  1e30f;
  cal->max_val = -1e30f;
  cal->scale   = 1.0f;
}

static inline void mlp_calibrator_update(mlp_act_quant_calibrator_t *cal,
                                          const float *x, int n) {
  for (int i = 0; i < n; i++) {
    if (x[i] < cal->min_val) cal->min_val = x[i];
    if (x[i] > cal->max_val) cal->max_val = x[i];
  }
}

static inline float mlp_calibrator_finalize(mlp_act_quant_calibrator_t *cal) {
  float abs_max = fabsf(cal->min_val) > fabsf(cal->max_val) ?
                  fabsf(cal->min_val) : fabsf(cal->max_val);
  cal->scale = abs_max / 127.0f;
  return cal->scale;
}

/* Quantize float activation vector to int8 */
static inline void mlp_quantize_activation(const float *x_f32, int8_t *x_q8,
                                            int n, float scale) {
  float inv = (scale > 1e-12f) ? (1.0f / scale) : 0.0f;
  for (int i = 0; i < n; i++) {
    int32_t q = (int32_t)roundf(x_f32[i] * inv);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    x_q8[i] = (int8_t)q;
  }
}

/* Dequantize int8 activation vector to float */
static inline void mlp_dequantize_activation(const int8_t *x_q8, float *x_f32,
                                              int n, float scale) {
  for (int i = 0; i < n; i++)
    x_f32[i] = (float)x_q8[i] * scale;
}

/* ------------------------------------------------------------------ */
/* Full int8 linear forward (requires int8 inputs)                     */
/* output = bias + scale_w * scale_x * int32_acc                      */
/* ------------------------------------------------------------------ */

static inline void mlp_linear_fwd_int8_full(const int8_t *x_q8, float x_scale,
                                             const int8_t *w_q8, const float *bias,
                                             const float *w_scales,
                                             int n_out, int n_in,
                                             float *out) {
  for (int r = 0; r < n_out; r++) {
    int32_t acc = mlp_dot_int8_raw(w_q8 + r * n_in, x_q8, n_in);
    out[r] = (float)acc * w_scales[r] * x_scale + (bias ? bias[r] : 0.0f);
  }
}

#endif /* MLP_QUANTIZE_H */
