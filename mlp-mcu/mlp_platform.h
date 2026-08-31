/*
 * mlp_platform.h — SIMD/platform abstraction layer
 *
 * Compile-time dispatch to the optimal backend for the target platform.
 * Zero runtime overhead: all selection happens via #ifdef preprocessor.
 *
 * Backend priority:
 *   1. x86 AVX2/FMA  (PC training, benchmarking)
 *   2. ARM Cortex-M FPU  (NEON + CMSIS-DSP)
 *   3. ARM Cortex-M fixed-point  (q15, no FPU)
 *   4. Xtensa (ESP32)
 *   5. Pure C scalar fallback
 *
 * Every platform must implement:
 *   - MLP_DOT16(a, b)          -> float: 16-wide dot product
 *   - MLP_DOT32(a, b)          -> float: 32-wide dot product
 *   - MLP_DOT64(a, b)          -> float: 64-wide dot product
 *   - MLP_DOT128(a, b)         -> float: 128-wide dot product
 *   - MLP_DOT_GENERIC(a, b, n) -> float: arbitrary-width dot product
 *   - MLP_LINEAR_FWD(x, w, bias, n_out, n_in, out) -> void
 *   - MLP_TANHF(x)             -> float: tanh activation
 *   - MLP_SIGMOIDF(x)          -> float: sigmoid activation
 *   - MLP_RMSNORM_FWD(x, n, out) -> float: RMS normalization
 *   - MLP_SOFTMAX_FWD(logits, n, probs) -> void
 *   - MLP_FEXPF(x)             -> float: fast exp (approximate)
 */

#ifndef MLP_PLATFORM_H
#define MLP_PLATFORM_H

/* ================================================================== */
/* Platform Detection                                                  */
/* ================================================================== */

#if defined(__AVX2__) || (defined(__x86_64__) && defined(__FMA__))
  #define MLP_PLATFORM_X86_AVX2

#elif defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_7M__) || \
      defined(__ARM_ARCH_8M_MAIN__) || defined(__ARM_ARCH_8M_BASE__)
  #if defined(__VFP_FP__) || defined(__FPU_PRESENT) || defined(__ARM_FP)
    #define MLP_PLATFORM_ARM_F32
  #else
    #define MLP_PLATFORM_ARM_Q15
  #endif

#elif defined(__XTENSA__) || defined(ESP_PLATFORM)
  #define MLP_PLATFORM_XTENSA

#else
  #define MLP_PLATFORM_SCALAR
#endif

/* ================================================================== */
/* Backend Includes                                                    */
/* ================================================================== */

#ifdef MLP_PLATFORM_X86_AVX2
  #include "mlp_platform_avx2.h"
#endif

#ifdef MLP_PLATFORM_ARM_F32
  #include "mlp_platform_armf32.h"
#endif

#ifdef MLP_PLATFORM_ARM_Q15
  #include "mlp_platform_q15.h"
#endif

/* ================================================================== */
/* Xtensa (ESP32) — inline scalar + xt-optimized math                  */
/* ================================================================== */

#ifdef MLP_PLATFORM_XTENSA
  #include <math.h>
  #include <xtensa/config/core-isa.h>

  static inline float mlp_dot16_xtensa(const float *a, const float *b) {
    float s = 0.0f;
    for (int i = 0; i < 16; i++) s += a[i] * b[i];
    return s;
  }
  static inline float mlp_dot32_xtensa(const float *a, const float *b) {
    float s = 0.0f;
    for (int i = 0; i < 32; i++) s += a[i] * b[i];
    return s;
  }
  static inline float mlp_dot64_xtensa(const float *a, const float *b) {
    float s = 0.0f;
    for (int i = 0; i < 64; i++) s += a[i] * b[i];
    return s;
  }
  static inline float mlp_dot128_xtensa(const float *a, const float *b) {
    float s = 0.0f;
    for (int i = 0; i < 128; i++) s += a[i] * b[i];
    return s;
  }
  static inline float mlp_dot_generic_xtensa(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
  }
  static inline float mlp_fexpf_xtensa(float x) {
    union { float f; int i; } u;
    u.i = (int)(12102203.1615614f * x * 1.4426950408f) + 1065353216;
    return u.f;
  }
  static inline float mlp_tanhf_xtensa(float x) { return tanhf(x); }
  static inline float mlp_sigmoidf_xtensa(float x) {
    return 0.5f * (1.0f + tanhf(0.5f * x));
  }
  static inline float mlp_rmsnorm_fwd_xtensa(const float *x, int n, float *out) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    float sc = 1.0f / sqrtf(sum / (float)n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = x[i] * sc;
    return sc;
  }
  static inline void mlp_softmax_fwd_xtensa(const float *logits, int n, float *probs) {
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { probs[i] = mlp_fexpf_xtensa(logits[i] - mx); sum += probs[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) probs[i] *= inv;
  }
  static inline void mlp_linear_fwd_xtensa(const float *x, const float *w,
                                            const float *bias, int n_out, int n_in, float *out) {
    for (int r = 0; r < n_out; r++) {
      float acc = 0.0f;
      for (int j = 0; j < n_in; j++) acc += w[r * n_in + j] * x[j];
      out[r] = acc + (bias ? bias[r] : 0.0f);
    }
  }
#endif

/* ================================================================== */
/* Scalar Fallback                                                     */
/* ================================================================== */

#ifdef MLP_PLATFORM_SCALAR
  #include <math.h>

  static inline float mlp_dot16_scalar(const float *a, const float *b) {
    float s = 0.0f; for (int i = 0; i < 16; i++) s += a[i] * b[i]; return s;
  }
  static inline float mlp_dot32_scalar(const float *a, const float *b) {
    float s = 0.0f; for (int i = 0; i < 32; i++) s += a[i] * b[i]; return s;
  }
  static inline float mlp_dot64_scalar(const float *a, const float *b) {
    float s = 0.0f; for (int i = 0; i < 64; i++) s += a[i] * b[i]; return s;
  }
  static inline float mlp_dot128_scalar(const float *a, const float *b) {
    float s = 0.0f; for (int i = 0; i < 128; i++) s += a[i] * b[i]; return s;
  }
  static inline float mlp_dot_generic_scalar(const float *a, const float *b, int n) {
    float s = 0.0f; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s;
  }
  static inline float mlp_fexpf_scalar(float x) {
    return expf(x);  /* libm exp */
  }
  static inline float mlp_tanhf_scalar(float x) { return tanhf(x); }
  static inline float mlp_sigmoidf_scalar(float x) {
    return 1.0f / (1.0f + expf(-x));
  }
  static inline float mlp_rmsnorm_fwd_scalar(const float *x, int n, float *out) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    float sc = 1.0f / sqrtf(sum / (float)n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = x[i] * sc;
    return sc;
  }
  static inline void mlp_softmax_fwd_scalar(const float *logits, int n, float *probs) {
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { probs[i] = expf(logits[i] - mx); sum += probs[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) probs[i] *= inv;
  }
  static inline void mlp_linear_fwd_scalar(const float *x, const float *w,
                                            const float *bias, int n_out, int n_in, float *out) {
    for (int r = 0; r < n_out; r++) {
      float acc = 0.0f;
      for (int j = 0; j < n_in; j++) acc += w[r * n_in + j] * x[j];
      out[r] = acc + (bias ? bias[r] : 0.0f);
    }
  }
#endif

/* ================================================================== */
/* Unified Dispatch Macros                                             */
/* All platform backends provide identically-named inline functions.   */
/* The macros below are thin wrappers that resolve at compile time.    */
/* ================================================================== */

#ifdef MLP_PLATFORM_X86_AVX2
  #define MLP_DOT16(a,b)           mlp_dot16_avx(a,b)
  #define MLP_DOT32(a,b)           mlp_dot32_avx(a,b)
  #define MLP_DOT64(a,b)           mlp_dot64_avx(a,b)
  #define MLP_DOT128(a,b)          mlp_dot128_avx(a,b)
  #define MLP_DOT_GENERIC(a,b,n)   mlp_dot_generic_avx(a,b,n)
  #define MLP_LINEAR_FWD(x,w,bo,no,ni,o) mlp_linear_fwd_avx(x,w,bo,no,ni,o)
  #define MLP_TANHF(x)             mlp_tanhf_avx(x)
  #define MLP_SIGMOIDF(x)          mlp_sigmoidf_avx(x)
  #define MLP_RMSNORM_FWD(x,n,o)   mlp_rmsnorm_fwd_avx(x,n,o)
  #define MLP_SOFTMAX_FWD(l,n,p)   mlp_softmax_fwd_avx(l,n,p)
  #define MLP_FEXPF(x)             mlp_fexpf_avx(x)

#elif defined(MLP_PLATFORM_ARM_F32)
  #define MLP_DOT16(a,b)           mlp_dot16_armf32(a,b)
  #define MLP_DOT32(a,b)           mlp_dot32_armf32(a,b)
  #define MLP_DOT64(a,b)           mlp_dot64_armf32(a,b)
  #define MLP_DOT128(a,b)          mlp_dot128_armf32(a,b)
  #define MLP_DOT_GENERIC(a,b,n)   mlp_dot_generic_armf32(a,b,n)
  #define MLP_LINEAR_FWD(x,w,bo,no,ni,o) mlp_linear_fwd_armf32(x,w,bo,no,ni,o)
  #define MLP_TANHF(x)             mlp_tanhf_armf32(x)
  #define MLP_SIGMOIDF(x)          mlp_sigmoidf_armf32(x)
  #define MLP_RMSNORM_FWD(x,n,o)   mlp_rmsnorm_fwd_armf32(x,n,o)
  #define MLP_SOFTMAX_FWD(l,n,p)   mlp_softmax_fwd_armf32(l,n,p)
  #define MLP_FEXPF(x)             mlp_fexpf_arm(x)

#elif defined(MLP_PLATFORM_ARM_Q15)
  /* q15 backend uses hybrid mode: float activation, q15 weights */
  #define MLP_DOT16(a,b)           mlp_dot16_q15_hybrid(a,b,1.0f)
  #define MLP_DOT32(a,b)           mlp_dot_generic_q15_raw(a,b,32)
  #define MLP_DOT64(a,b)           mlp_dot64_q15_hybrid(a,b,1.0f)
  #define MLP_DOT128(a,b)          mlp_dot_generic_q15_raw(a,b,128)
  #define MLP_DOT_GENERIC(a,b,n)   mlp_dot_generic_q15_raw(a,b,n)
  #define MLP_LINEAR_FWD(x,w,bo,no,ni,o) mlp_linear_fwd_q15(x,w,bo,1.0f,no,ni,o)
  #define MLP_TANHF(x)             mlp_tanhf_q15(x)
  #define MLP_SIGMOIDF(x)          mlp_sigmoidf_q15(x)
  #define MLP_RMSNORM_FWD(x,n,o)   mlp_rmsnorm_fwd_q15(x,n,o)
  #define MLP_SOFTMAX_FWD(l,n,p)   mlp_softmax_fwd_q15(l,n,p)
  #define MLP_FEXPF(x)             expf(x)

#elif defined(MLP_PLATFORM_XTENSA)
  #define MLP_DOT16(a,b)           mlp_dot16_xtensa(a,b)
  #define MLP_DOT32(a,b)           mlp_dot32_xtensa(a,b)
  #define MLP_DOT64(a,b)           mlp_dot64_xtensa(a,b)
  #define MLP_DOT128(a,b)          mlp_dot128_xtensa(a,b)
  #define MLP_DOT_GENERIC(a,b,n)   mlp_dot_generic_xtensa(a,b,n)
  #define MLP_LINEAR_FWD(x,w,bo,no,ni,o) mlp_linear_fwd_xtensa(x,w,bo,no,ni,o)
  #define MLP_TANHF(x)             mlp_tanhf_xtensa(x)
  #define MLP_SIGMOIDF(x)          mlp_sigmoidf_xtensa(x)
  #define MLP_RMSNORM_FWD(x,n,o)   mlp_rmsnorm_fwd_xtensa(x,n,o)
  #define MLP_SOFTMAX_FWD(l,n,p)   mlp_softmax_fwd_xtensa(l,n,p)
  #define MLP_FEXPF(x)             mlp_fexpf_xtensa(x)

#else /* MLP_PLATFORM_SCALAR */
  #define MLP_DOT16(a,b)           mlp_dot16_scalar(a,b)
  #define MLP_DOT32(a,b)           mlp_dot32_scalar(a,b)
  #define MLP_DOT64(a,b)           mlp_dot64_scalar(a,b)
  #define MLP_DOT128(a,b)          mlp_dot128_scalar(a,b)
  #define MLP_DOT_GENERIC(a,b,n)   mlp_dot_generic_scalar(a,b,n)
  #define MLP_LINEAR_FWD(x,w,bo,no,ni,o) mlp_linear_fwd_scalar(x,w,bo,no,ni,o)
  #define MLP_TANHF(x)             mlp_tanhf_scalar(x)
  #define MLP_SIGMOIDF(x)          mlp_sigmoidf_scalar(x)
  #define MLP_RMSNORM_FWD(x,n,o)   mlp_rmsnorm_fwd_scalar(x,n,o)
  #define MLP_SOFTMAX_FWD(l,n,p)   mlp_softmax_fwd_scalar(l,n,p)
  #define MLP_FEXPF(x)             mlp_fexpf_scalar(x)
#endif

/* ================================================================== */
/* Activation enum (for model.h configuration)                         */
/* ================================================================== */

typedef enum {
  MLP_ACT_NONE = 0,    /* Identity (no-op, for regression output) */
  MLP_ACT_RELU,        /* ReLU: max(0, x) */
  MLP_ACT_RELU2,       /* ReLU^2: (x>0) ? x*x : 0 */
  MLP_ACT_TANH,        /* Tanh: [-1, +1] */
  MLP_ACT_SIGMOID,     /* Sigmoid: [0, +1] */
  MLP_ACT_CLIPPED_RELU /* ReLU with upper bound */
} mlp_act_t;

#endif /* MLP_PLATFORM_H */
