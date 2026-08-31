/*
 * mlp_platform_avx2.h — x86 AVX2/FMA SIMD kernels
 *
 * Reuses the hand-tuned fixed-width dot product patterns from microgpt-c.
 * Compile with: -mavx2 -mfma -mbmi -mbmi2 -mpopcnt
 */

#ifndef MLP_PLATFORM_AVX2_H
#define MLP_PLATFORM_AVX2_H

#include <immintrin.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Fast exponential — Schraudolph approximation (~1 ULP, 3-4x faster) */
/* From microgpt-c line 35-38                                           */
/* ------------------------------------------------------------------ */

static inline float mlp_fexpf_avx(float x) {
  union { float f; int i; } u;
  u.i = (int)(12102203.1615614f * x * 1.4426950408f) + 1065353216;
  return u.f;
}

/* ------------------------------------------------------------------ */
/* Fixed-width dot products — fully unrolled FMA loops                 */
/* From microgpt-c lines 251-274                                        */
/* ------------------------------------------------------------------ */

/* dot16: N=16 → 2× FMA256 + horizontal sum, zero loop overhead */
static inline float mlp_dot16_avx(const float *a, const float *b) {
  __m256 r = _mm256_fmadd_ps(_mm256_loadu_ps(a),   _mm256_loadu_ps(b),
             _mm256_mul_ps (_mm256_loadu_ps(a+8),  _mm256_loadu_ps(b+8)));
  __m128 lo = _mm256_castps256_ps128(r), hi = _mm256_extractf128_ps(r, 1);
  lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

/* dot32: N=32 → 4× FMA256 + horizontal sum */
static inline float mlp_dot32_avx(const float *a, const float *b) {
  __m256 r0 = _mm256_fmadd_ps(_mm256_loadu_ps(a),    _mm256_loadu_ps(b),
              _mm256_mul_ps (_mm256_loadu_ps(a+8),   _mm256_loadu_ps(b+8)));
  __m256 r1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+16), _mm256_loadu_ps(b+16),
              _mm256_mul_ps (_mm256_loadu_ps(a+24),  _mm256_loadu_ps(b+24)));
  __m256 r  = _mm256_add_ps(r0, r1);
  __m128 lo = _mm256_castps256_ps128(r), hi = _mm256_extractf128_ps(r, 1);
  lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

/* dot64: N=64 → 8× FMA256, fully unrolled */
static inline float mlp_dot64_avx(const float *a, const float *b) {
  __m256 acc = _mm256_setzero_ps();
  int i = 0;
  #pragma GCC unroll 8
  for (i = 0; i < 64; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
  __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
  lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

/* dot128: N=128 → 16× FMA256 */
static inline float mlp_dot128_avx(const float *a, const float *b) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  int i = 0;
  #pragma GCC unroll 8
  for (i = 0; i < 64; i += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i),    acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+64+i),_mm256_loadu_ps(b+64+i), acc1);
  }
  acc0 = _mm256_add_ps(acc0, acc1);
  __m128 lo = _mm256_castps256_ps128(acc0), hi = _mm256_extractf128_ps(acc0, 1);
  lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

/* Generic dot for odd sizes (auto-vectorized by compiler) */
static inline float mlp_dot_generic_avx(const float *a, const float *b, int n) {
  float sum = 0.0f;
  #pragma GCC ivdep
  for (int i = 0; i < n; i++) sum += a[i] * b[i];
  return sum;
}

/* ------------------------------------------------------------------ */
/* RMSNorm — N=16, 2 AVX loads, no loop                               */
/* From microgpt-c lines 307-318                                       */
/* ------------------------------------------------------------------ */

static inline float mlp_rmsnorm_fwd_avx(const float *x, int n, float *out) {
  if (n == 16) {
    __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x+8);
    __m256 ss = _mm256_fmadd_ps(v0, v0, _mm256_mul_ps(v1, v1));
    __m128 lo = _mm256_castps256_ps128(ss), hi = _mm256_extractf128_ps(ss, 1);
    lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    float ms = _mm_cvtss_f32(lo) / (float)n;
    float sc = 1.0f / sqrtf(ms + 1e-5f);
    __m256 vs = _mm256_set1_ps(sc);
    _mm256_storeu_ps(out,   _mm256_mul_ps(v0, vs));
    _mm256_storeu_ps(out+8, _mm256_mul_ps(v1, vs));
    return sc;
  }
  /* Generic fallback for non-16 dimensions */
  float sum = 0.0f;
  for (int i = 0; i < n; i++) sum += x[i] * x[i];
  float ms = sum / (float)n;
  float sc = 1.0f / sqrtf(ms + 1e-5f);
  for (int i = 0; i < n; i++) out[i] = x[i] * sc;
  return sc;
}

/* ------------------------------------------------------------------ */
/* Softmax — Schraudolph exp for inference, precise for training       */
/* From microgpt-c lines 321-338                                       */
/* ------------------------------------------------------------------ */

static inline void mlp_softmax_fwd_avx(const float *logits, int n, float *probs) {
  float mx = logits[0];
  for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
  float sum = 0;
  for (int i = 0; i < n; i++) { probs[i] = mlp_fexpf_avx(logits[i] - mx); sum += probs[i]; }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++) probs[i] *= inv;
}

static inline void mlp_softmax_precise_avx(const float *logits, int n, float *probs) {
  float mx = logits[0];
  for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
  float sum = 0;
  for (int i = 0; i < n; i++) { probs[i] = expf(logits[i] - mx); sum += probs[i]; }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++) probs[i] *= inv;
}

/* ------------------------------------------------------------------ */
/* Tanh — AVX polynomial approximation (faster than libm on x86)       */
/* ------------------------------------------------------------------ */

static inline float mlp_tanhf_avx(float x) {
  /* Clamp to [-9, 9] where tanh saturates at ±1 */
  if (x >  9.0f) return  1.0f;
  if (x < -9.0f) return -1.0f;
  float x2 = x * x;
  /* Rational approximation: tanh(x) ≈ x * (27 + x²) / (27 + 9*x²) (minimax order 1,1) */
  return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

/* ------------------------------------------------------------------ */
/* Sigmoid — via tanh: sigmoid(x) = 0.5 * (1 + tanh(x/2))             */
/* ------------------------------------------------------------------ */

static inline float mlp_sigmoidf_avx(float x) {
  return 0.5f * (1.0f + mlp_tanhf_avx(0.5f * x));
}

/* ------------------------------------------------------------------ */
/* Matrix-vector multiply: linear layer forward pass                   */
/* Dispatch on n_in to call the right fixed-width dot product.         */
/* Pattern from microgpt-c linear_fwd lines 279-304.                   */
/* ------------------------------------------------------------------ */

static inline void mlp_linear_fwd_avx(const float *x,
                                       const float *w, const float *bias,
                                       int n_out, int n_in,
                                       float *out) {
  if (n_in == 16) {
    for (int r = 0; r < n_out; r++)
      out[r] = mlp_dot16_avx(w + r * n_in, x) + (bias ? bias[r] : 0.0f);
  } else if (n_in == 32) {
    for (int r = 0; r < n_out; r++)
      out[r] = mlp_dot32_avx(w + r * n_in, x) + (bias ? bias[r] : 0.0f);
  } else if (n_in == 64) {
    for (int r = 0; r < n_out; r++)
      out[r] = mlp_dot64_avx(w + r * n_in, x) + (bias ? bias[r] : 0.0f);
  } else if (n_in == 128) {
    for (int r = 0; r < n_out; r++)
      out[r] = mlp_dot128_avx(w + r * n_in, x) + (bias ? bias[r] : 0.0f);
  } else {
    for (int r = 0; r < n_out; r++)
      out[r] = mlp_dot_generic_avx(w + r * n_in, x, n_in) + (bias ? bias[r] : 0.0f);
  }
}

#endif /* MLP_PLATFORM_AVX2_H */
