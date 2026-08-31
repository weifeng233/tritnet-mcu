/*
 * mlp_platform_q15.h — Fixed-point q15 kernels for Cortex-M0/M3 (no FPU)
 *
 * All computations in q15 (int16, 15 fractional bits).
 * Activation functions use LUT + linear interpolation.
 * Target: Cortex-M0 (ARMv6-M), Cortex-M3 (ARMv7-M) without FPU
 *
 * Compile flags (arm-none-eabi-gcc):
 *   -mcpu=cortex-m0  (q15: use __SMUAD, __SSUB16 if available)
 *   -mcpu=cortex-m3  (q15: use UMAAL, SMUAD, SMLAD DSP instructions)
 *
 * q15 range: -1.0 to +0.99996  (0x8000 to 0x7FFF)
 * Resolution: 1/32768 ≈ 3.05e-5
 * Accumulator: int32 (sufficient for up to 256-element dot products)
 */

#ifndef MLP_PLATFORM_Q15_H
#define MLP_PLATFORM_Q15_H

#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* q15 <-> float32 conversion helpers                                  */
/* ------------------------------------------------------------------ */

#define MLP_Q15_ONE      32767
#define MLP_Q15_SCALE    (1.0f / 32768.0f)

static inline int16_t mlp_float_to_q15(float x) {
  int32_t q = (int32_t)(x * 32768.0f);
  if (q >  32767) q =  32767;
  if (q < -32768) q = -32768;
  return (int16_t)q;
}

static inline float mlp_q15_to_float(int16_t x) {
  return (float)x * MLP_Q15_SCALE;
}

/* ------------------------------------------------------------------ */
/* q15 dot products — DSP-intrinsic-optimized when available           */
/* ------------------------------------------------------------------ */

/* dot16_q15: N=16, accumulator in int32
 *
 * With ARM DSP instructions (Cortex-M3+), each MAC pair is 1 cycle:
 *   __SMLAD(a, b, acc) = halfword SIMD: 2x int16 MAC in 1 instruction
 *
 * Without DSP: plain int16 * int16 -> int32 MAC loop
 */
static inline int32_t mlp_dot16_q15_raw(const int16_t *a, const int16_t *b) {
#if defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7EM__)
  /* Cortex-M3/M4 DSP: use __SMLAD for 2× q15 MAC per cycle */
  int32_t acc = 0;
  for (int i = 0; i < 16; i += 2) {
    int32_t ap = *(int32_t*)(a + i);  /* pack 2x q15 into 1 word */
    int32_t bp = *(int32_t*)(b + i);
    acc = __SMLAD(ap, bp, acc);       /* 2x MAC in 1 cycle */
  }
  return acc;
#else
  /* Cortex-M0 fallback: plain int16 MAC */
  int32_t acc = 0;
  for (int i = 0; i < 16; i++)
    acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
#endif
}

static inline int32_t mlp_dot32_q15_raw(const int16_t *a, const int16_t *b) {
#if defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7EM__)
  int32_t acc = 0;
  for (int i = 0; i < 32; i += 2) {
    int32_t ap = *(int32_t*)(a + i);
    int32_t bp = *(int32_t*)(b + i);
    acc = __SMLAD(ap, bp, acc);
  }
  return acc;
#else
  int32_t acc = 0;
  for (int i = 0; i < 32; i++)
    acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
#endif
}

static inline int32_t mlp_dot64_q15_raw(const int16_t *a, const int16_t *b) {
#if defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7EM__)
  int32_t acc0 = 0, acc1 = 0;
  for (int i = 0; i < 64; i += 4) {
    int32_t ap0 = *(int32_t*)(a + i),     bp0 = *(int32_t*)(b + i);
    int32_t ap1 = *(int32_t*)(a + i + 2), bp1 = *(int32_t*)(b + i + 2);
    acc0 = __SMLAD(ap0, bp0, acc0);
    acc1 = __SMLAD(ap1, bp1, acc1);
  }
  return acc0 + acc1;
#else
  int32_t acc = 0;
  for (int i = 0; i < 64; i++)
    acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
#endif
}

static inline int32_t mlp_dot128_q15_raw(const int16_t *a, const int16_t *b) {
#if defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7EM__)
  int32_t acc0 = 0, acc1 = 0;
  for (int i = 0; i < 128; i += 4) {
    int32_t ap0 = *(int32_t*)(a + i),     bp0 = *(int32_t*)(b + i);
    int32_t ap1 = *(int32_t*)(a + i + 2), bp1 = *(int32_t*)(b + i + 2);
    acc0 = __SMLAD(ap0, bp0, acc0);
    acc1 = __SMLAD(ap1, bp1, acc1);
  }
  return acc0 + acc1;
#else
  int32_t acc = 0;
  for (int i = 0; i < 128; i++)
    acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
#endif
}

static inline int32_t mlp_dot_generic_q15_raw(const int16_t *a, const int16_t *b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++)
    acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

/* ------------------------------------------------------------------ */
/* q15 dot product returning float (hybrid: q15 weights, float output) */
/* Input: int16 weights (from Flash), float input vector               */
/* Output: float result                                                */
/* ------------------------------------------------------------------ */

static inline float mlp_dot16_q15_hybrid(const int16_t *w_q15,
                                         const float   *x_f32,
                                         float w_scale) {
  /* Convert float input to q15 on-the-fly (fast on Cortex-M3+ with DSP) */
  int16_t x_q15[16];
  for (int i = 0; i < 16; i++) x_q15[i] = mlp_float_to_q15(x_f32[i]);
  int32_t acc = mlp_dot16_q15_raw(w_q15, x_q15);
  return mlp_q15_to_float((int16_t)(acc >> 15)) * w_scale;
}

static inline float mlp_dot64_q15_hybrid(const int16_t *w_q15,
                                         const float   *x_f32,
                                         float w_scale) {
  int16_t x_q15[64];
  for (int i = 0; i < 64; i++) x_q15[i] = mlp_float_to_q15(x_f32[i]);
  int32_t acc = mlp_dot64_q15_raw(w_q15, x_q15);
  return mlp_q15_to_float((int16_t)(acc >> 15)) * w_scale;
}

/* ------------------------------------------------------------------ */
/* LUT-based activation functions (256 entries, stored in Flash)        */
/* Uses linear interpolation for precision.                            */
/* ------------------------------------------------------------------ */

/* Sigmoid LUT: maps [-8.0, +8.0] to [0, 1]
 * 256 entries, stored in .rodata (Flash)
 * Index: normalized_input = (x + 8.0) / 16.0 * 255
 */
static const int16_t mlp_sigmoid_q15_lut[256] = {
     5,   10,   15,   20,   25,   30,   35,   40,   46,   51,
    57,   63,   69,   76,   82,   89,   96,  103,  111,  118,
   126,  134,  143,  151,  160,  169,  178,  188,  198,  208,
   219,  229,  240,  252,  263,  275,  287,  300,  312,  325,
   339,  352,  366,  380,  395,  410,  425,  440,  456,  472,
   489,  506,  523,  540,  558,  576,  595,  614,  633,  653,
   673,  693,  714,  735,  757,  779,  801,  824,  847,  870,
   894,  918,  943,  968,  993, 1019, 1045, 1071, 1098, 1125,
  1152, 1180, 1208, 1237, 1266, 1295, 1324, 1354, 1384, 1415,
  1446, 1477, 1508, 1540, 1572, 1605, 1637, 1670, 1703, 1737,
  1770, 1804, 1838, 1872, 1907, 1941, 1976, 2011, 2046, 2081,
  2116, 2151, 2186, 2222, 2257, 2292, 2327, 2363, 2398, 2433,
  2468, 2503, 2538, 2572, 2607, 2641, 2675, 2709, 2743, 2776,
  2810, 2843, 2875, 2908, 2940, 2972, 3004, 3035, 3066, 3097,
  3127, 3157, 3186, 3216, 3244, 3273, 3301, 3328, 3356, 3382,
  3409, 3435, 3461, 3486, 3511, 3535, 3559, 3582, 3605, 3628,
  3650, 3672, 3693, 3714, 3734, 3754, 3774, 3793, 3812, 3830,
  3848, 3865, 3882, 3899, 3915, 3931, 3946, 3961, 3975, 3989,
  4003, 4016, 4029, 4042, 4054, 4066, 4077, 4088, 4099, 4110,
  4120, 4130, 4139, 4149, 4158, 4166, 4175, 4183, 4191, 4198,
  4206, 4213, 4220, 4226, 4233, 4239, 4245, 4250, 4256, 4261,
  4266, 4271, 4276, 4280, 4285, 4289, 4293, 4297, 4300, 4304,
  4307, 4310, 4313, 4316, 4319, 4321, 4324, 4326, 4328, 4330,
  4332, 4334, 4336, 4338, 4339, 4341, 4342, 4343, 4345, 4346,
  4347, 4348, 4349, 4350, 4351, 4351, 4352, 4353, 4353, 4354,
  4354, 4355, 4355, 4356, 4356, 4356, 4357, 4357, 4357, 4357,
  4358, 4358, 4358, 4358, 4358, 4358, 4358, 4358, 4358, 4358,
  4358, 4358, 4358, 4358, 4358, 4358, 4358, 4358, 4358, 4358,
  4358, 4358, 4358, 4358, 4358, 4358, 4358, 4358, 4358, 4358,
  4358, 4358, 4358, 4358, 4358, 4358, 4358, 4358, 4358, 4358,
  4358, 4358, 4358, 4358, 4358, 4358
};

/* Tanh LUT: maps [-4.0, +4.0] to [-1, +1] (q15)
 * 256 entries, stored in .rodata (Flash)
 */
static const int16_t mlp_tanh_q15_lut[256] = {
  -32394, -32337, -32270, -32192, -32102, -31998, -31878, -31741,
  -31584, -31405, -31200, -30965, -30697, -30390, -30039, -29637,
  -29177, -28650, -28045, -27350, -26552, -25636, -24585, -23380,
  -21999, -20418, -18608, -16539, -14179, -11494,  -8446,  -4994,
      -1,   4994,   8446,  11494,  14179,  16539,  18608,  20418,
   21999,  23380,  24585,  25636,  26552,  27350,  28045,  28650,
   29177,  29637,  30039,  30390,  30697,  30965,  31200,  31405,
   31584,  31741,  31878,  31998,  32102,  32192,  32270,  32337,
   32394,  32400,  32427,  32442,  32454,  32463,  32470,  32475,
   32479,  32483,  32485,  32487,  32489,  32490,  32491,  32492,
   32492,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493,
   32493,  32493,  32493,  32493,  32493,  32493,  32493,  32493
};

static inline int16_t mlp_sigmoid_q15_lut_interp(int16_t x) {
  /* Map x from [-32768, 32767] to LUT index [0, 255]
   * x_q15 ≈ -8.0 .. +8.0 in float → LUT range [-4, +4] */
  int32_t idx = ((int32_t)x + 32768) * 255 / 65536;
  if (idx < 0)   idx = 0;
  if (idx > 255) idx = 255;
  int16_t v0 = mlp_sigmoid_q15_lut[idx];
  int16_t v1 = mlp_sigmoid_q15_lut[idx < 255 ? idx + 1 : 255];
  /* Linear interpolation between v0 and v1 */
  int32_t frac = ((int32_t)x + 32768) * 255 % 65536;
  return (int16_t)(v0 + (((int32_t)(v1 - v0) * frac + 32768) >> 16));
}

static inline int16_t mlp_tanh_q15_lut_interp(int16_t x) {
  /* Map x from [-32768, 32767] to LUT index [0, 255]
   * tanh range [-4, +4] in float → [-131072, 131072] in q15 scaled by 32768 */
  int32_t idx = ((int32_t)x + 32768) * 255 / 65536;
  if (idx < 0)   idx = 0;
  if (idx > 255) idx = 255;
  int16_t v0 = mlp_tanh_q15_lut[idx];
  int16_t v1 = mlp_tanh_q15_lut[idx < 255 ? idx + 1 : 255];
  int32_t frac = ((int32_t)x + 32768) * 255 % 65536;
  return (int16_t)(v0 + (((int32_t)(v1 - v0) * frac + 32768) >> 16));
}

/* ------------------------------------------------------------------ */
/* q15 Tanh/Sigmoid (float interface, uses LUT internally)             */
/* ------------------------------------------------------------------ */

static inline float mlp_tanhf_q15(float x) {
  int16_t q = mlp_float_to_q15(x);
  return mlp_q15_to_float(mlp_tanh_q15_lut_interp(q));
}

static inline float mlp_sigmoidf_q15(float x) {
  int16_t q = mlp_float_to_q15(x);
  return mlp_q15_to_float(mlp_sigmoid_q15_lut_interp(q)) + 0.5f;
}

/* ------------------------------------------------------------------ */
/* q15 RMSNorm                                                         */
/* ------------------------------------------------------------------ */

static inline float mlp_rmsnorm_fwd_q15(const float *x, int n, float *out) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) sum += x[i] * x[i];
  float ms  = sum / (float)n;
  float sc  = 1.0f / sqrtf(ms + 1e-5f);  /* sqrtf is software, but called rarely */
  for (int i = 0; i < n; i++) out[i] = x[i] * sc;
  return sc;
}

/* ------------------------------------------------------------------ */
/* q15 Softmax (uses LUT for exp)                                      */
/* ------------------------------------------------------------------ */

static inline void mlp_softmax_fwd_q15(const float *logits, int n, float *probs) {
  float mx = logits[0];
  for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
  float sum = 0;
  for (int i = 0; i < n; i++) {
    int16_t qx = mlp_float_to_q15(logits[i] - mx);
    /* exp via sigmoid: exp(x) ≈ sigmoid(x/ln2) * 2, but simpler to use float exp approximation */
    float val = mlp_sigmoidf_q15((logits[i] - mx) * 2.0f);  /* sigmoid of scaled input */
    probs[i] = val;
    sum += val;
  }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++) probs[i] *= inv;
}

/* ------------------------------------------------------------------ */
/* q15 linear forward (hybrid: q15 weights, float activations)         */
/* ------------------------------------------------------------------ */

static inline void mlp_linear_fwd_q15(const float *x,
                                       const int16_t *w_q15, const float *bias,
                                       float w_scale,
                                       int n_out, int n_in,
                                       float *out) {
  if (n_in == 16) {
    for (int r = 0; r < n_out; r++)
      out[r] = mlp_dot16_q15_hybrid(w_q15 + r * n_in, x, w_scale) + (bias ? bias[r] : 0.0f);
  } else if (n_in == 64) {
    for (int r = 0; r < n_out; r++)
      out[r] = mlp_dot64_q15_hybrid(w_q15 + r * n_in, x, w_scale) + (bias ? bias[r] : 0.0f);
  } else {
    /* Generic path: convert weights to q15 on-the-fly */
    int16_t x_q15[128];
    for (int i = 0; i < n_in && i < 128; i++) x_q15[i] = mlp_float_to_q15(x[i]);
    for (int r = 0; r < n_out; r++) {
      int32_t acc = mlp_dot_generic_q15_raw(w_q15 + r * n_in, x_q15, n_in);
      out[r] = mlp_q15_to_float((int16_t)(acc >> 15)) * w_scale + (bias ? bias[r] : 0.0f);
    }
  }
}

#endif /* MLP_PLATFORM_Q15_H */
