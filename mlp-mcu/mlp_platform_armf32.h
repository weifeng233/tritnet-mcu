/*
 * mlp_platform_armf32.h — Cortex-M4/M7 FPU (NEON) SIMD kernels
 *
 * Uses ARM CMSIS-DSP when available, otherwise manual NEON intrinsics.
 * Target: Cortex-M4F, Cortex-M7 (STM32F4, STM32H7, etc.)
 *
 * Compile flags (arm-none-eabi-gcc):
 *   -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard
 * or:
 *   -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard
 */

#ifndef MLP_PLATFORM_ARMF32_H
#define MLP_PLATFORM_ARMF32_H

#include <math.h>

/* CMSIS-DSP availability (when supported by toolchain) */
#if __has_include("arm_math.h")
  #include "arm_math.h"
  #define MLP_HAS_CMSIS_DSP  1
#endif

/* ------------------------------------------------------------------ */
/* Fast exponential — Schraudolph (same algorithm as AVX2 version)     */
/* ------------------------------------------------------------------ */

static inline float mlp_fexpf_arm(float x) {
  union { float f; int i; } u;
  u.i = (int)(12102203.1615614f * x * 1.4426950408f) + 1065353216;
  return u.f;
}

/* ------------------------------------------------------------------ */
/* Dot products — manual NEON intrinsics (4-wide float32)               */
/* ------------------------------------------------------------------ */

/* dot16: N=16 → 4× NEON float32x4 (8 MUL + 7 ADD = 15 ops) */
static inline float mlp_dot16_armf32(const float *a, const float *b) {
#if defined(__ARM_NEON) || defined(__aarch64__)
  float32x4_t sum0 = vmulq_f32(vld1q_f32(a),    vld1q_f32(b));
  float32x4_t sum1 = vmulq_f32(vld1q_f32(a+4),  vld1q_f32(b+4));
  float32x4_t sum2 = vmulq_f32(vld1q_f32(a+8),  vld1q_f32(b+8));
  float32x4_t sum3 = vmulq_f32(vld1q_f32(a+12), vld1q_f32(b+12));
  sum0 = vaddq_f32(sum0, sum1);
  sum2 = vaddq_f32(sum2, sum3);
  sum0 = vaddq_f32(sum0, sum2);
  /* Horizontal sum: 2× pairwise add + extract */
  float32x2_t lo = vget_low_f32(sum0);
  float32x2_t hi = vget_high_f32(sum0);
  lo = vpadd_f32(lo, hi);
  lo = vpadd_f32(lo, lo);
  return vget_lane_f32(lo, 0);
#elif defined(MLP_HAS_CMSIS_DSP)
  float result;
  arm_dot_prod_f32(a, b, 16, &result);
  return result;
#else
  float sum = 0.0f;
  for (int i = 0; i < 16; i++) sum += a[i] * b[i];
  return sum;
#endif
}

/* dot32: N=32 → 8× NEON float32x4 */
static inline float mlp_dot32_armf32(const float *a, const float *b) {
#if defined(__ARM_NEON) || defined(__aarch64__)
  float32x4_t sum0 = vmulq_f32(vld1q_f32(a),    vld1q_f32(b));
  float32x4_t sum1 = vmulq_f32(vld1q_f32(a+4),  vld1q_f32(b+4));
  float32x4_t sum2 = vmulq_f32(vld1q_f32(a+8),  vld1q_f32(b+8));
  float32x4_t sum3 = vmulq_f32(vld1q_f32(a+12), vld1q_f32(b+12));
  float32x4_t sum4 = vmulq_f32(vld1q_f32(a+16), vld1q_f32(b+16));
  float32x4_t sum5 = vmulq_f32(vld1q_f32(a+20), vld1q_f32(b+20));
  float32x4_t sum6 = vmulq_f32(vld1q_f32(a+24), vld1q_f32(b+24));
  float32x4_t sum7 = vmulq_f32(vld1q_f32(a+28), vld1q_f32(b+28));
  sum0 = vaddq_f32(sum0, sum1); sum2 = vaddq_f32(sum2, sum3);
  sum4 = vaddq_f32(sum4, sum5); sum6 = vaddq_f32(sum6, sum7);
  sum0 = vaddq_f32(sum0, sum2); sum4 = vaddq_f32(sum4, sum6);
  sum0 = vaddq_f32(sum0, sum4);
  float32x2_t lo = vget_low_f32(sum0);
  float32x2_t hi = vget_high_f32(sum0);
  lo = vpadd_f32(lo, hi);
  lo = vpadd_f32(lo, lo);
  return vget_lane_f32(lo, 0);
#elif defined(MLP_HAS_CMSIS_DSP)
  float result;
  arm_dot_prod_f32(a, b, 32, &result);
  return result;
#else
  float sum = 0.0f;
  for (int i = 0; i < 32; i++) sum += a[i] * b[i];
  return sum;
#endif
}

/* dot64: N=64 → 16× NEON float32x4, loop unrolled 4x */
static inline float mlp_dot64_armf32(const float *a, const float *b) {
#if defined(__ARM_NEON) || defined(__aarch64__)
  float32x4_t acc0 = vmulq_f32(vld1q_f32(a),    vld1q_f32(b));
  float32x4_t acc1 = vmulq_f32(vld1q_f32(a+4),  vld1q_f32(b+4));
  float32x4_t acc2 = vmulq_f32(vld1q_f32(a+8),  vld1q_f32(b+8));
  float32x4_t acc3 = vmulq_f32(vld1q_f32(a+12), vld1q_f32(b+12));
  for (int i = 16; i < 64; i += 16) {
    acc0 = vmlaq_f32(acc0, vld1q_f32(a+i),    vld1q_f32(b+i));
    acc1 = vmlaq_f32(acc1, vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
    acc2 = vmlaq_f32(acc2, vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
    acc3 = vmlaq_f32(acc3, vld1q_f32(a+i+12), vld1q_f32(b+i+12));
  }
  acc0 = vaddq_f32(acc0, acc1); acc2 = vaddq_f32(acc2, acc3);
  acc0 = vaddq_f32(acc0, acc2);
  float32x2_t lo = vget_low_f32(acc0);
  float32x2_t hi = vget_high_f32(acc0);
  lo = vpadd_f32(lo, hi);
  lo = vpadd_f32(lo, lo);
  return vget_lane_f32(lo, 0);
#elif defined(MLP_HAS_CMSIS_DSP)
  float result;
  arm_dot_prod_f32(a, b, 64, &result);
  return result;
#else
  float sum = 0.0f;
  for (int i = 0; i < 64; i++) sum += a[i] * b[i];
  return sum;
#endif
}

/* dot128: N=128 → double accumulator */
static inline float mlp_dot128_armf32(const float *a, const float *b) {
#if defined(__ARM_NEON) || defined(__aarch64__)
  float32x4_t acc0 = vmulq_f32(vld1q_f32(a),    vld1q_f32(b));
  float32x4_t acc1 = vmulq_f32(vld1q_f32(a+4),  vld1q_f32(b+4));
  float32x4_t acc2 = vmulq_f32(vld1q_f32(a+8),  vld1q_f32(b+8));
  float32x4_t acc3 = vmulq_f32(vld1q_f32(a+12), vld1q_f32(b+12));
  for (int i = 16; i < 128; i += 16) {
    acc0 = vmlaq_f32(acc0, vld1q_f32(a+i),    vld1q_f32(b+i));
    acc1 = vmlaq_f32(acc1, vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
    acc2 = vmlaq_f32(acc2, vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
    acc3 = vmlaq_f32(acc3, vld1q_f32(a+i+12), vld1q_f32(b+i+12));
  }
  acc0 = vaddq_f32(acc0, acc1); acc2 = vaddq_f32(acc2, acc3);
  acc0 = vaddq_f32(acc0, acc2);
  float32x2_t lo = vget_low_f32(acc0);
  float32x2_t hi = vget_high_f32(acc0);
  lo = vpadd_f32(lo, hi);
  lo = vpadd_f32(lo, lo);
  return vget_lane_f32(lo, 0);
#elif defined(MLP_HAS_CMSIS_DSP)
  float result;
  arm_dot_prod_f32(a, b, 128, &result);
  return result;
#else
  float sum = 0.0f;
  for (int i = 0; i < 128; i++) sum += a[i] * b[i];
  return sum;
#endif
}

static inline float mlp_dot_generic_armf32(const float *a, const float *b, int n) {
#if defined(MLP_HAS_CMSIS_DSP)
  float result;
  arm_dot_prod_f32(a, b, (uint32_t)n, &result);
  return result;
#else
  float sum = 0.0f;
  for (int i = 0; i < n; i++) sum += a[i] * b[i];
  return sum;
#endif
}

/* ------------------------------------------------------------------ */
/* RMSNorm — ARM NEON float32                                          */
/* ------------------------------------------------------------------ */

static inline float mlp_rmsnorm_fwd_armf32(const float *x, int n, float *out) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) sum += x[i] * x[i];
  float ms  = sum / (float)n;
  float sc  = 1.0f / sqrtf(ms + 1e-5f);
#if defined(__ARM_NEON) || defined(__aarch64__)
  float32x4_t vs = vdupq_n_f32(sc);
  int i = 0;
  for (i = 0; i <= n - 4; i += 4)
    vst1q_f32(out + i, vmulq_f32(vld1q_f32(x + i), vs));
  for (; i < n; i++) out[i] = x[i] * sc;
#else
  for (int i = 0; i < n; i++) out[i] = x[i] * sc;
#endif
  return sc;
}

/* ------------------------------------------------------------------ */
/* Softmax — Schraudolph fast exp                                      */
/* ------------------------------------------------------------------ */

static inline void mlp_softmax_fwd_armf32(const float *logits, int n, float *probs) {
  float mx = logits[0];
  for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
  float sum = 0;
  for (int i = 0; i < n; i++) { probs[i] = mlp_fexpf_arm(logits[i] - mx); sum += probs[i]; }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++) probs[i] *= inv;
}

static inline void mlp_softmax_precise_armf32(const float *logits, int n, float *probs) {
  float mx = logits[0];
  for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
  float sum = 0;
  for (int i = 0; i < n; i++) { probs[i] = expf(logits[i] - mx); sum += probs[i]; }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++) probs[i] *= inv;
}

/* ------------------------------------------------------------------ */
/* Tanh/Sigmoid — math.h for Cortex-M (hardware FPU)                  */
/* ------------------------------------------------------------------ */

static inline float mlp_tanhf_armf32(float x) {
  return tanhf(x);  /* hardware-accelerated on Cortex-M4F/M7 */
}

static inline float mlp_sigmoidf_armf32(float x) {
  return 0.5f * (1.0f + tanhf(0.5f * x));
}

/* ------------------------------------------------------------------ */
/* Matrix-vector linear forward                                        */
/* ------------------------------------------------------------------ */

static inline void mlp_linear_fwd_armf32(const float *x,
                                          const float *w, const float *bias,
                                          int n_out, int n_in,
                                          float *out) {
  if (n_in == 16) {
    for (int r = 0; r < n_out; r++)
      out[r] = mlp_dot16_armf32(w + r * n_in, x) + (bias ? bias[r] : 0.0f);
  } else if (n_in == 32) {
    for (int r = 0; r < n_out; r++)
      out[r] = mlp_dot32_armf32(w + r * n_in, x) + (bias ? bias[r] : 0.0f);
  } else if (n_in == 64) {
    for (int r = 0; r < n_out; r++)
      out[r] = mlp_dot64_armf32(w + r * n_in, x) + (bias ? bias[r] : 0.0f);
  } else if (n_in == 128) {
    for (int r = 0; r < n_out; r++)
      out[r] = mlp_dot128_armf32(w + r * n_in, x) + (bias ? bias[r] : 0.0f);
  } else {
    for (int r = 0; r < n_out; r++)
      out[r] = mlp_dot_generic_armf32(w + r * n_in, x, n_in) + (bias ? bias[r] : 0.0f);
  }
}

#endif /* MLP_PLATFORM_ARMF32_H */
