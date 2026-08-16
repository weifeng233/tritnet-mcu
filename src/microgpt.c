#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  #define MG_SIMD_AVX2 1
  #define MG_SIMD_NAME "AVX2"
#elif defined(__aarch64__) || defined(_M_ARM64)
  #define MG_SIMD_NEON 1
  #define MG_SIMD_NAME "NEON"
#else
  #error "microGPT-C requires x86-64 (AVX2) or AArch64 (NEON)"
#endif

#ifndef _MSC_VER
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("O3,unroll-loops")
#endif
#ifdef MG_SIMD_AVX2
#pragma GCC target("avx2,fma,bmi,bmi2,popcnt")
#endif
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  static double now_s(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
  }
#else
  #include <time.h>
  static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
  }
#endif
#ifdef _WIN32
  #include <malloc.h>
  #define MG_ALIGNED_ALLOC(a, n) _aligned_malloc((n), (a))
  #define MG_ALIGNED_FREE(p)     _aligned_free(p)
#else
  #define MG_ALIGNED_ALLOC(a, n) aligned_alloc((a), (n))
  #define MG_ALIGNED_FREE(p)     free(p)
#endif

#ifdef MG_SIMD_AVX2
  #include <immintrin.h>
#else
  #include <arm_neon.h>
#endif

static __attribute__((always_inline)) inline float fexpf(float x) {
  union { float f; int i; } u;
  u.i = (int)(12102203.1615614f * x * 1.4426950408f) + 1065353216;
  return u.f;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static unsigned long long rng_state = 42;

static unsigned long long rng_next(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}

static double rng_uniform(void) {
  return (rng_next() >> 11) * (1.0 / 9007199254740992.0);
}

static float rng_gauss(float mean, float std) {
  double u1 = rng_uniform(), u2 = rng_uniform();
  if (u1 < 1e-30)
    u1 = 1e-30;
  return mean + std * (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}

static void shuffle_ints(int *arr, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = (int)(rng_uniform() * (i + 1));
    int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

#define MAX_DOCS 85000
#define MAX_DOC_LEN 512
#define MAX_CHARS 128

static char docs[MAX_DOCS][MAX_DOC_LEN];
static int num_docs = 0;

static void load_dataset(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", filename);
    exit(1);
  }
  char line[256];
  while (fgets(line, sizeof(line), f) && num_docs < MAX_DOCS) {
    int len = (int)strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
      line[--len] = 0;
    if (len > 0) {
      strncpy(docs[num_docs], line, MAX_DOC_LEN - 1);
      docs[num_docs][MAX_DOC_LEN - 1] = 0;
      num_docs++;
    }
  }
  fclose(f);
}

static char uchars_arr[MAX_CHARS];
static int vocab_size, BOS, num_uchars = 0;

static int char_to_id(char c) {
  for (int i = 0; i < num_uchars; i++)
    if (uchars_arr[i] == c)
      return i;
  return -1;
}

static int cmp_char(const void *a, const void *b) {
  return *(const char *)a - *(const char *)b;
}

static void build_tokenizer(void) {
  int seen[256] = {0};
  for (int d = 0; d < num_docs; d++)
    for (int i = 0; docs[d][i]; i++)
      seen[(unsigned char)docs[d][i]] = 1;
  for (int i = 0; i < 256; i++)
    if (seen[i])
      uchars_arr[num_uchars++] = (char)i;
  qsort(uchars_arr, num_uchars, sizeof(char), cmp_char);
  BOS = num_uchars;
  vocab_size = num_uchars + 1;
}

#define N_EMBD 16
#define N_HEAD 4
#define N_LAYER 1
#define BLOCK_SIZE 16
#define HEAD_DIM (N_EMBD / N_HEAD)
#define MLP_DIM (4 * N_EMBD)

static float *wte, *d_wte;
static float *wpe, *d_wpe;
static float *lm_head, *d_lm_head;

static float *attn_wq[N_LAYER], *d_attn_wq[N_LAYER];
static float *attn_wk[N_LAYER], *d_attn_wk[N_LAYER];
static float *attn_wv[N_LAYER], *d_attn_wv[N_LAYER];
static float *attn_wo[N_LAYER], *d_attn_wo[N_LAYER];
static float *mlp_fc1[N_LAYER], *d_mlp_fc1[N_LAYER];
static float *mlp_fc2[N_LAYER], *d_mlp_fc2[N_LAYER];

static float *adam_m_wte, *adam_v_wte;
static float *adam_m_wpe, *adam_v_wpe;
static float *adam_m_lm, *adam_v_lm;
static float *adam_m_wq[N_LAYER], *adam_v_wq[N_LAYER];
static float *adam_m_wk[N_LAYER], *adam_v_wk[N_LAYER];
static float *adam_m_wv[N_LAYER], *adam_v_wv[N_LAYER];
static float *adam_m_wo[N_LAYER], *adam_v_wo[N_LAYER];
static float *adam_m_fc1[N_LAYER], *adam_v_fc1[N_LAYER];
static float *adam_m_fc2[N_LAYER], *adam_v_fc2[N_LAYER];

static int num_params = 0;

static float *make_param(int size, float std) {
  float *p = (float *)calloc(size, sizeof(float));
  for (int i = 0; i < size; i++)
    p[i] = rng_gauss(0, std);
  num_params += size;
  return p;
}

static float *make_zero(int size) {
  return (float *)calloc(size, sizeof(float));
}

static void init_params(void) {
  int es = vocab_size * N_EMBD, ps = BLOCK_SIZE * N_EMBD;
  int as = N_EMBD * N_EMBD, ms = MLP_DIM * N_EMBD;
  wte = make_param(es, 0.02f);
  d_wte = make_zero(es);
  adam_m_wte = make_zero(es);
  adam_v_wte = make_zero(es);
  wpe = make_param(ps, 0.02f);
  d_wpe = make_zero(ps);
  adam_m_wpe = make_zero(ps);
  adam_v_wpe = make_zero(ps);
  lm_head = make_param(es, 0.02f);
  d_lm_head = make_zero(es);
  adam_m_lm = make_zero(es);
  adam_v_lm = make_zero(es);
  for (int i = 0; i < N_LAYER; i++) {
    attn_wq[i] = make_param(as, 0.02f);
    d_attn_wq[i] = make_zero(as);
    adam_m_wq[i] = make_zero(as);
    adam_v_wq[i] = make_zero(as);
    attn_wk[i] = make_param(as, 0.02f);
    d_attn_wk[i] = make_zero(as);
    adam_m_wk[i] = make_zero(as);
    adam_v_wk[i] = make_zero(as);
    attn_wv[i] = make_param(as, 0.02f);
    d_attn_wv[i] = make_zero(as);
    adam_m_wv[i] = make_zero(as);
    adam_v_wv[i] = make_zero(as);
    attn_wo[i] = make_param(as, 0.0f);
    d_attn_wo[i] = make_zero(as);
    adam_m_wo[i] = make_zero(as);
    adam_v_wo[i] = make_zero(as);
    mlp_fc1[i] = make_param(ms, 0.02f);
    d_mlp_fc1[i] = make_zero(ms);
    adam_m_fc1[i] = make_zero(ms);
    adam_v_fc1[i] = make_zero(ms);
    mlp_fc2[i] = make_param(ms, 0.0f);
    d_mlp_fc2[i] = make_zero(ms);
    adam_m_fc2[i] = make_zero(ms);
    adam_v_fc2[i] = make_zero(ms);
  }
  printf("num params: %d\n", num_params);
}

typedef struct {
  float x_embed[N_EMBD];
  float rms_scale_init;
  float x_in[N_LAYER][N_EMBD];
  float xn_attn[N_LAYER][N_EMBD];
  float rms_scale_attn[N_LAYER];
  float q[N_LAYER][N_EMBD];
  float aw[N_LAYER][N_HEAD][BLOCK_SIZE];
  float attn_out[N_LAYER][N_EMBD];
  float x_mid[N_LAYER][N_EMBD];
  float xn_mlp[N_LAYER][N_EMBD];
  float rms_scale_mlp[N_LAYER];
  float mlp_pre[N_LAYER][MLP_DIM];
  float mlp_post[N_LAYER][MLP_DIM];
  float x_out[N_EMBD];
} PosActs;

static PosActs saved[BLOCK_SIZE];
static float saved_probs[BLOCK_SIZE][MAX_CHARS + 1];

static float kv_keys[N_LAYER][BLOCK_SIZE][N_EMBD];
static float kv_vals[N_LAYER][BLOCK_SIZE][N_EMBD];
static float dk_accum[N_LAYER][BLOCK_SIZE][N_EMBD];
static float dv_accum[N_LAYER][BLOCK_SIZE][N_EMBD];

#ifdef MG_SIMD_AVX2

static __attribute__((always_inline)) inline float dot16(const float *a, const float *b) {
  __m256 r = _mm256_fmadd_ps(_mm256_loadu_ps(a),   _mm256_loadu_ps(b),
             _mm256_mul_ps (_mm256_loadu_ps(a+8),  _mm256_loadu_ps(b+8)));
  __m128 lo = _mm256_castps256_ps128(r), hi = _mm256_extractf128_ps(r, 1);
  lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

static __attribute__((always_inline)) inline float dot4(const float *a, const float *b) {
  __m128 r = _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
  r = _mm_hadd_ps(r, r); r = _mm_hadd_ps(r, r);
  return _mm_cvtss_f32(r);
}

static __attribute__((always_inline)) inline float dot64(const float *a, const float *b) {
  __m256 acc = _mm256_setzero_ps();
  for (int i = 0; i < 64; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
  __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
  lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

#else

static __attribute__((always_inline)) inline float dot16(const float *a, const float *b) {
  float32x4_t acc0 = vmulq_f32(vld1q_f32(a),    vld1q_f32(b));
  float32x4_t acc1 = vmulq_f32(vld1q_f32(a+4),  vld1q_f32(b+4));
  acc0 = vfmaq_f32(acc0, vld1q_f32(a+8),  vld1q_f32(b+8));
  acc1 = vfmaq_f32(acc1, vld1q_f32(a+12), vld1q_f32(b+12));
  return vaddvq_f32(vaddq_f32(acc0, acc1));
}

static __attribute__((always_inline)) inline float dot4(const float *a, const float *b) {
  return vaddvq_f32(vmulq_f32(vld1q_f32(a), vld1q_f32(b)));
}

static __attribute__((always_inline)) inline float dot64(const float *a, const float *b) {
  float32x4_t acc0 = vmulq_f32(vld1q_f32(a),    vld1q_f32(b));
  float32x4_t acc1 = vmulq_f32(vld1q_f32(a+4),  vld1q_f32(b+4));
  float32x4_t acc2 = vmulq_f32(vld1q_f32(a+8),  vld1q_f32(b+8));
  float32x4_t acc3 = vmulq_f32(vld1q_f32(a+12), vld1q_f32(b+12));
  for (int i = 16; i < 64; i += 16) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a+i),    vld1q_f32(b+i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(a+i+12), vld1q_f32(b+i+12));
  }
  return vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
}

#endif

static __attribute__((always_inline)) inline void linear_fwd(const float *restrict x,
                                       const float *restrict w,
                                       int nout, int nin,
                                       float *restrict out) {
  if (nin == N_EMBD && nout == N_EMBD) {
    for (int r = 0; r < 16; r++) out[r] = dot16(w + r*16, x);
  } else if (nin == N_EMBD && nout == MLP_DIM) {
    for (int r = 0; r < 64; r++) out[r] = dot16(w + r*16, x);
  } else if (nin == MLP_DIM && nout == N_EMBD) {
    for (int r = 0; r < 16; r++) out[r] = dot64(w + r*64, x);
  } else {
#ifdef MG_SIMD_AVX2
    __m256 vx0 = _mm256_loadu_ps(x), vx1 = _mm256_loadu_ps(x+8);
    int r = 0;
    for (; r <= nout-4; r += 4) {
      __m256 a0=_mm256_fmadd_ps(_mm256_loadu_ps(w+r*16),    vx0,_mm256_mul_ps(_mm256_loadu_ps(w+r*16+8),    vx1));
      __m256 a1=_mm256_fmadd_ps(_mm256_loadu_ps(w+(r+1)*16),vx0,_mm256_mul_ps(_mm256_loadu_ps(w+(r+1)*16+8),vx1));
      __m256 a2=_mm256_fmadd_ps(_mm256_loadu_ps(w+(r+2)*16),vx0,_mm256_mul_ps(_mm256_loadu_ps(w+(r+2)*16+8),vx1));
      __m256 a3=_mm256_fmadd_ps(_mm256_loadu_ps(w+(r+3)*16),vx0,_mm256_mul_ps(_mm256_loadu_ps(w+(r+3)*16+8),vx1));
      __m256 t0=_mm256_hadd_ps(a0,a1), t1=_mm256_hadd_ps(a2,a3);
      __m256 t2=_mm256_hadd_ps(t0,t1);
      __m128 lo=_mm256_castps256_ps128(t2), hi=_mm256_extractf128_ps(t2,1);
      _mm_storeu_ps(out+r, _mm_add_ps(lo,hi));
    }
#else
    const float32x4_t x0 = vld1q_f32(x),    x1 = vld1q_f32(x+4);
    const float32x4_t x2 = vld1q_f32(x+8),  x3 = vld1q_f32(x+12);
    int r = 0;
    for (; r <= nout-4; r += 4) {
      const float *w0 = w + r*16, *w1 = w0+16, *w2 = w1+16, *w3 = w2+16;
      float32x4_t s0 = vmulq_f32(vld1q_f32(w0), x0);
      float32x4_t s1 = vmulq_f32(vld1q_f32(w1), x0);
      float32x4_t s2 = vmulq_f32(vld1q_f32(w2), x0);
      float32x4_t s3 = vmulq_f32(vld1q_f32(w3), x0);
      s0 = vfmaq_f32(s0, vld1q_f32(w0+4), x1);
      s1 = vfmaq_f32(s1, vld1q_f32(w1+4), x1);
      s2 = vfmaq_f32(s2, vld1q_f32(w2+4), x1);
      s3 = vfmaq_f32(s3, vld1q_f32(w3+4), x1);
      s0 = vfmaq_f32(s0, vld1q_f32(w0+8), x2);
      s1 = vfmaq_f32(s1, vld1q_f32(w1+8), x2);
      s2 = vfmaq_f32(s2, vld1q_f32(w2+8), x2);
      s3 = vfmaq_f32(s3, vld1q_f32(w3+8), x2);
      s0 = vfmaq_f32(s0, vld1q_f32(w0+12), x3);
      s1 = vfmaq_f32(s1, vld1q_f32(w1+12), x3);
      s2 = vfmaq_f32(s2, vld1q_f32(w2+12), x3);
      s3 = vfmaq_f32(s3, vld1q_f32(w3+12), x3);
      out[r]   = vaddvq_f32(s0);
      out[r+1] = vaddvq_f32(s1);
      out[r+2] = vaddvq_f32(s2);
      out[r+3] = vaddvq_f32(s3);
    }
#endif
    for (; r < nout; r++) out[r] = dot16(w + r*16, x);
  }
}

static __attribute__((always_inline)) inline float rmsnorm_fwd(const float *x, int n, float *out) {
#ifdef MG_SIMD_AVX2
  __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x+8);
  __m256 ss = _mm256_fmadd_ps(v0, v0, _mm256_mul_ps(v1, v1));
  __m128 lo = _mm256_castps256_ps128(ss), hi = _mm256_extractf128_ps(ss, 1);
  lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
  float ms = _mm_cvtss_f32(lo) / (float)n;
  float sc = 1.0f / sqrtf(ms + 1e-5f);
  __m256 vs = _mm256_set1_ps(sc);
  _mm256_storeu_ps(out,   _mm256_mul_ps(v0, vs));
  _mm256_storeu_ps(out+8, _mm256_mul_ps(v1, vs));
#else
  float32x4_t v0 = vld1q_f32(x),   v1 = vld1q_f32(x+4);
  float32x4_t v2 = vld1q_f32(x+8), v3 = vld1q_f32(x+12);
  float32x4_t ss = vmulq_f32(v0, v0);
  ss = vfmaq_f32(ss, v1, v1);
  ss = vfmaq_f32(ss, v2, v2);
  ss = vfmaq_f32(ss, v3, v3);
  float ms = vaddvq_f32(ss) / (float)n;
  float sc = 1.0f / sqrtf(ms + 1e-5f);
  float32x4_t vs = vdupq_n_f32(sc);
  vst1q_f32(out,    vmulq_f32(v0, vs));
  vst1q_f32(out+4,  vmulq_f32(v1, vs));
  vst1q_f32(out+8,  vmulq_f32(v2, vs));
  vst1q_f32(out+12, vmulq_f32(v3, vs));
#endif
  return sc;
}

static __attribute__((always_inline)) inline void softmax_fwd_precise(const float *logits, int n, float *probs) {
  float mx = logits[0];
  for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
  float sum = 0;
  for (int i = 0; i < n; i++) { probs[i] = expf(logits[i] - mx); sum += probs[i]; }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++) probs[i] *= inv;
}

static inline void linear_bwd_x(const float *restrict w,
                                const float *restrict dout, int nout, int nin,
                                float *restrict dx) {
  for (int c = 0; c < nin; c++) {
    float s = 0;
    for (int r = 0; r < nout; r++)
      s += dout[r] * w[r * nin + c];
    dx[c] += s;
  }
}

static inline void linear_bwd_w(const float *restrict x,
                                const float *restrict dout, int nout, int nin,
                                float *restrict dw) {
  for (int r = 0; r < nout; r++) {
    float dr = dout[r];
    float *dwr = dw + r * nin;
    for (int c = 0; c < nin; c++)
      dwr[c] += dr * x[c];
  }
}

static inline void rmsnorm_bwd(const float *x, float scale, const float *dout,
                               int n, float *dx) {
  float dot = 0;
  for (int i = 0; i < n; i++)
    dot += dout[i] * x[i];
  float coeff = scale * scale * scale / n;
  for (int i = 0; i < n; i++)
    dx[i] += scale * dout[i] - coeff * x[i] * dot;
}

static void gpt_forward(int token_id, int pos_id, float *logits_out,
                        PosActs *act) {
  float x[N_EMBD], tmp[MLP_DIM > N_EMBD ? MLP_DIM : N_EMBD];

  for (int i = 0; i < N_EMBD; i++)
    x[i] = wte[token_id * N_EMBD + i] + wpe[pos_id * N_EMBD + i];
  memcpy(act->x_embed, x, sizeof(x));

  act->rms_scale_init = rmsnorm_fwd(x, N_EMBD, x);

  for (int li = 0; li < N_LAYER; li++) {
    memcpy(act->x_in[li], x, sizeof(x));

    float xn[N_EMBD];
    act->rms_scale_attn[li] = rmsnorm_fwd(x, N_EMBD, xn);
    memcpy(act->xn_attn[li], xn, sizeof(xn));

    float q[N_EMBD], k[N_EMBD], v[N_EMBD];
    linear_fwd(xn, attn_wq[li], N_EMBD, N_EMBD, q);
    linear_fwd(xn, attn_wk[li], N_EMBD, N_EMBD, k);
    linear_fwd(xn, attn_wv[li], N_EMBD, N_EMBD, v);
    memcpy(act->q[li], q, sizeof(q));

    memcpy(kv_keys[li][pos_id], k, sizeof(k));
    memcpy(kv_vals[li][pos_id], v, sizeof(v));
    int seq_len = pos_id + 1;
    float scale = 1.0f / sqrtf((float)N_EMBD / (float)N_HEAD);

    float ao[N_EMBD];
    for (int h = 0; h < N_HEAD; h++) {
      int hs = h * HEAD_DIM;
      float al[BLOCK_SIZE];
      for (int tt = 0; tt < seq_len; tt++)
        al[tt] = dot4(q + hs, kv_keys[li][tt] + hs) * scale;
      float mx = al[0];
      for (int tt = 1; tt < seq_len; tt++)
        if (al[tt] > mx)
          mx = al[tt];
      float sm = 0;
      for (int tt = 0; tt < seq_len; tt++) {
        al[tt] = fexpf(al[tt] - mx);
        sm += al[tt];
      }
      float inv = 1.0f / sm;
      for (int tt = 0; tt < seq_len; tt++)
        al[tt] *= inv;
      for (int tt = 0; tt < seq_len; tt++)
        act->aw[li][h][tt] = al[tt];
      for (int j = 0; j < HEAD_DIM; j++) {
        float s = 0;
        for (int tt = 0; tt < seq_len; tt++)
          s += al[tt] * kv_vals[li][tt][hs + j];
        ao[hs + j] = s;
      }
    }
    memcpy(act->attn_out[li], ao, sizeof(ao));

    linear_fwd(ao, attn_wo[li], N_EMBD, N_EMBD, tmp);
    for (int i = 0; i < N_EMBD; i++)
      x[i] = tmp[i] + act->x_in[li][i];
    memcpy(act->x_mid[li], x, sizeof(x));

    float xn_m[N_EMBD];
    act->rms_scale_mlp[li] = rmsnorm_fwd(x, N_EMBD, xn_m);
    memcpy(act->xn_mlp[li], xn_m, sizeof(xn_m));

    float h1[MLP_DIM];
    linear_fwd(xn_m, mlp_fc1[li], MLP_DIM, N_EMBD, h1);
    memcpy(act->mlp_pre[li], h1, MLP_DIM * sizeof(float));

    float h2[MLP_DIM];
    for (int i = 0; i < MLP_DIM; i++)
      h2[i] = h1[i] > 0 ? h1[i] * h1[i] : 0;
    memcpy(act->mlp_post[li], h2, MLP_DIM * sizeof(float));

    linear_fwd(h2, mlp_fc2[li], N_EMBD, MLP_DIM, tmp);
    for (int i = 0; i < N_EMBD; i++)
      x[i] = tmp[i] + act->x_mid[li][i];
  }

  memcpy(act->x_out, x, sizeof(x));
  linear_fwd(x, lm_head, vocab_size, N_EMBD, logits_out);
}

#define LM_PAD_MAX ((((MAX_CHARS + 1) + 3) / 4) * 4)
#define ATTN_SCALE (0.5f)

static int lm_pad = LM_PAD_MAX;

#ifdef MG_SIMD_NEON

static __attribute__((always_inline)) inline float32x4_t vfexpq(float32x4_t x) {
  float32x4_t y = vmulq_n_f32(x, 12102203.1615614f * 1.4426950408f);
  int32x4_t i = vaddq_s32(vcvtq_s32_f32(y), vdupq_n_s32(1065353216));
  return vreinterpretq_f32_s32(i);
}

#else

static __attribute__((always_inline)) inline __m128 vfexpq(__m128 x) {
  __m128 y = _mm_mul_ps(x, _mm_set1_ps(12102203.1615614f * 1.4426950408f));
  __m128i i = _mm_add_epi32(_mm_cvttps_epi32(y), _mm_set1_epi32(1065353216));
  return _mm_castsi128_ps(i);
}

static __attribute__((always_inline)) inline float hsum128(__m128 v) {
  v = _mm_hadd_ps(v, v);
  v = _mm_hadd_ps(v, v);
  return _mm_cvtss_f32(v);
}

static __attribute__((always_inline)) inline float hmax128(__m128 v) {
  v = _mm_max_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)));
  v = _mm_max_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
  return _mm_cvtss_f32(v);
}

#endif

static int sample_logits(float *restrict logits, int n, int npad, float inv_t) {
  float p[LM_PAD_MAX];
  float sum;
#ifdef MG_SIMD_NEON

  for (int i = n; i < npad; i++) logits[i] = logits[0];

  float32x4_t m = vld1q_f32(logits);
  for (int i = 4; i < npad; i += 4) m = vmaxq_f32(m, vld1q_f32(logits + i));
  const float32x4_t vit = vdupq_n_f32(inv_t);
  const float32x4_t vmx = vdupq_n_f32(vmaxvq_f32(m) * inv_t);

  float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
  int i = 0;
  for (; i + 8 <= npad; i += 8) {
    float32x4_t e0 = vfexpq(vsubq_f32(vmulq_f32(vld1q_f32(logits + i),     vit), vmx));
    float32x4_t e1 = vfexpq(vsubq_f32(vmulq_f32(vld1q_f32(logits + i + 4), vit), vmx));
    vst1q_f32(p + i, e0); vst1q_f32(p + i + 4, e1);
    s0 = vaddq_f32(s0, e0); s1 = vaddq_f32(s1, e1);
  }
  for (; i < npad; i += 4) {
    float32x4_t e0 = vfexpq(vsubq_f32(vmulq_f32(vld1q_f32(logits + i), vit), vmx));
    vst1q_f32(p + i, e0);
    s0 = vaddq_f32(s0, e0);
  }
  sum = vaddvq_f32(vaddq_f32(s0, s1)) - (float)(npad - n) * p[0];
#else

  for (int i = n; i < npad; i++) logits[i] = logits[0];

  __m128 m = _mm_loadu_ps(logits);
  for (int i = 4; i < npad; i += 4) m = _mm_max_ps(m, _mm_loadu_ps(logits + i));
  const __m128 vit = _mm_set1_ps(inv_t);
  const __m128 vmx = _mm_set1_ps(hmax128(m) * inv_t);

  __m128 s0 = _mm_setzero_ps(), s1 = s0;
  int i = 0;
  for (; i + 8 <= npad; i += 8) {
    __m128 e0 = vfexpq(_mm_sub_ps(_mm_mul_ps(_mm_loadu_ps(logits + i),     vit), vmx));
    __m128 e1 = vfexpq(_mm_sub_ps(_mm_mul_ps(_mm_loadu_ps(logits + i + 4), vit), vmx));
    _mm_storeu_ps(p + i, e0); _mm_storeu_ps(p + i + 4, e1);
    s0 = _mm_add_ps(s0, e0); s1 = _mm_add_ps(s1, e1);
  }
  for (; i < npad; i += 4) {
    __m128 e0 = vfexpq(_mm_sub_ps(_mm_mul_ps(_mm_loadu_ps(logits + i), vit), vmx));
    _mm_storeu_ps(p + i, e0);
    s0 = _mm_add_ps(s0, e0);
  }
  sum = hsum128(_mm_add_ps(s0, s1)) - (float)(npad - n) * p[0];
#endif
  float r = (float)rng_uniform() * sum, cum = 0;
  for (int i = 0; i < n; i++) { cum += p[i]; if (r < cum) return i; }
  return n - 1;
}

static __attribute__((aligned(64))) float iw_q[N_LAYER][N_EMBD * N_EMBD];
static __attribute__((aligned(64))) float iw_k[N_LAYER][N_EMBD * N_EMBD];
static __attribute__((aligned(64))) float iw_v[N_LAYER][N_EMBD * N_EMBD];
static __attribute__((aligned(64))) float iw_o[N_LAYER][N_EMBD * N_EMBD];
static __attribute__((aligned(64))) float iw_fc1[N_LAYER][N_EMBD * MLP_DIM];
static __attribute__((aligned(64))) float iw_fc2[N_LAYER][MLP_DIM * N_EMBD];
static __attribute__((aligned(64))) float iw_lm[N_EMBD * LM_PAD_MAX];

static void pack_t(const float *src, float *dst, int nout, int nin) {
  for (int r = 0; r < nout; r++)
    for (int c = 0; c < nin; c++)
      dst[(size_t)c * nout + r] = src[(size_t)r * nin + c];
}

typedef struct {
  float xin[N_EMBD];
  float q[N_EMBD];
  float k[N_EMBD];
  float v[N_EMBD];
} PreTok;

static PreTok *pretok;
static void build_pretok(void);

static void infer_pack_weights(void) {
  lm_pad = ((vocab_size + 3) / 4) * 4;
  for (int li = 0; li < N_LAYER; li++) {
    pack_t(attn_wq[li], iw_q[li],  N_EMBD,  N_EMBD);
    pack_t(attn_wk[li], iw_k[li],  N_EMBD,  N_EMBD);
    pack_t(attn_wv[li], iw_v[li],  N_EMBD,  N_EMBD);
    pack_t(attn_wo[li], iw_o[li],  N_EMBD,  N_EMBD);

    for (int bi = 0; bi < MLP_DIM / 16; bi++)
      for (int c = 0; c < N_EMBD; c++)
        for (int r = 0; r < 16; r++)
          iw_fc1[li][bi * 256 + c * 16 + r] =
              mlp_fc1[li][(size_t)(bi * 16 + r) * N_EMBD + c];
    pack_t(mlp_fc2[li], iw_fc2[li], N_EMBD,  MLP_DIM);
  }
  memset(iw_lm, 0, sizeof(iw_lm));
  for (int r = 0; r < vocab_size; r++)
    for (int c = 0; c < N_EMBD; c++)
      iw_lm[(size_t)c * lm_pad + r] = lm_head[(size_t)r * N_EMBD + c];
  build_pretok();
}

#ifdef MG_SIMD_NEON

#define ACC(p0, p1, p2, p3, wp, xv, L)                            \
  p0 = vfmaq_laneq_f32(p0, vld1q_f32((wp)),      xv, L);          \
  p1 = vfmaq_laneq_f32(p1, vld1q_f32((wp) + 4),  xv, L);          \
  p2 = vfmaq_laneq_f32(p2, vld1q_f32((wp) + 8),  xv, L);          \
  p3 = vfmaq_laneq_f32(p3, vld1q_f32((wp) + 12), xv, L)

#define ACC4(p0, p1, p2, p3, w, ldw, xv)                          \
  ACC(p0, p1, p2, p3, (w),               xv, 0);                  \
  ACC(p0, p1, p2, p3, (w) + (ldw),       xv, 1);                  \
  ACC(p0, p1, p2, p3, (w) + 2 * (ldw),   xv, 2);                  \
  ACC(p0, p1, p2, p3, (w) + 3 * (ldw),   xv, 3)

typedef struct { float32x4_t v0, v1, v2, v3; } vec16;

static __attribute__((always_inline)) inline
vec16 mv16_blk16v(vec16 xr, const float *restrict wcol, int ldw) {
  float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
  float32x4_t b0 = a0, b1 = a0, b2 = a0, b3 = a0;
  const size_t s = (size_t)ldw;
  ACC4(a0, a1, a2, a3, wcol,            s, xr.v0);
  ACC4(b0, b1, b2, b3, wcol + 4  * s,   s, xr.v1);
  ACC4(a0, a1, a2, a3, wcol + 8  * s,   s, xr.v2);
  ACC4(b0, b1, b2, b3, wcol + 12 * s,   s, xr.v3);
  vec16 r = { vaddq_f32(a0, b0), vaddq_f32(a1, b1),
              vaddq_f32(a2, b2), vaddq_f32(a3, b3) };
  return r;
}

#define ACC2(p0, p1, wp, xv, L)                                   \
  p0 = vfmaq_laneq_f32(p0, vld1q_f32((wp)),     xv, L);           \
  p1 = vfmaq_laneq_f32(p1, vld1q_f32((wp) + 4), xv, L)
#define ACC2x4(p0, p1, w, ldw, xv)                                \
  ACC2(p0, p1, (w),             xv, 0);                           \
  ACC2(p0, p1, (w) + (ldw),     xv, 1);                           \
  ACC2(p0, p1, (w) + 2 * (ldw), xv, 2);                           \
  ACC2(p0, p1, (w) + 3 * (ldw), xv, 3)
#define ACC1(p0, wp, xv, L) p0 = vfmaq_laneq_f32(p0, vld1q_f32((wp)), xv, L)
#define ACC1x4(p0, w, ldw, xv)                                    \
  ACC1(p0, (w),             xv, 0);                               \
  ACC1(p0, (w) + (ldw),     xv, 1);                               \
  ACC1(p0, (w) + 2 * (ldw), xv, 2);                               \
  ACC1(p0, (w) + 3 * (ldw), xv, 3)

static __attribute__((always_inline)) inline
void mv16_blk8(const float *restrict x, const float *restrict wcol,
               int ldw, float *restrict out) {
  float32x4_t a0 = vdupq_n_f32(0), a1 = a0, b0 = a0, b1 = a0;
  const size_t s = (size_t)ldw;
  ACC2x4(a0, a1, wcol,            s, vld1q_f32(x));
  ACC2x4(b0, b1, wcol + 4  * s,   s, vld1q_f32(x + 4));
  ACC2x4(a0, a1, wcol + 8  * s,   s, vld1q_f32(x + 8));
  ACC2x4(b0, b1, wcol + 12 * s,   s, vld1q_f32(x + 12));
  vst1q_f32(out,     vaddq_f32(a0, b0));
  vst1q_f32(out + 4, vaddq_f32(a1, b1));
}

static __attribute__((always_inline)) inline
void mv16_blk4(const float *restrict x, const float *restrict wcol,
               int ldw, float *restrict out) {
  float32x4_t a = vdupq_n_f32(0), b = a;
  const size_t s = (size_t)ldw;
  ACC1x4(a, wcol,            s, vld1q_f32(x));
  ACC1x4(b, wcol + 4  * s,   s, vld1q_f32(x + 4));
  ACC1x4(a, wcol + 8  * s,   s, vld1q_f32(x + 8));
  ACC1x4(b, wcol + 12 * s,   s, vld1q_f32(x + 12));
  vst1q_f32(out, vaddq_f32(a, b));
}

static __attribute__((always_inline)) inline
vec16 mv16_blk16v4(vec16 xr, const float *restrict wcol, int ldw) {
  float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
  float32x4_t b0 = a0, b1 = a0, b2 = a0, b3 = a0;
  float32x4_t c0 = a0, c1 = a0, c2 = a0, c3 = a0;
  float32x4_t d0 = a0, d1 = a0, d2 = a0, d3 = a0;
  const size_t s = (size_t)ldw;
  ACC4(a0, a1, a2, a3, wcol,          s, xr.v0);
  ACC4(b0, b1, b2, b3, wcol + 4  * s, s, xr.v1);
  ACC4(c0, c1, c2, c3, wcol + 8  * s, s, xr.v2);
  ACC4(d0, d1, d2, d3, wcol + 12 * s, s, xr.v3);
  vec16 r = { vaddq_f32(vaddq_f32(a0, b0), vaddq_f32(c0, d0)),
              vaddq_f32(vaddq_f32(a1, b1), vaddq_f32(c1, d1)),
              vaddq_f32(vaddq_f32(a2, b2), vaddq_f32(c2, d2)),
              vaddq_f32(vaddq_f32(a3, b3), vaddq_f32(c3, d3)) };
  return r;
}

static __attribute__((always_inline)) inline
vec16 mv16_blk16r4(const float *restrict x, const float *restrict wcol, int ldw) {
  vec16 xr = { vld1q_f32(x), vld1q_f32(x + 4),
               vld1q_f32(x + 8), vld1q_f32(x + 12) };
  return mv16_blk16v4(xr, wcol, ldw);
}

static __attribute__((always_inline)) inline
vec16 mv16_blk16r(const float *restrict x, const float *restrict wcol, int ldw) {
  vec16 xr = { vld1q_f32(x), vld1q_f32(x + 4),
               vld1q_f32(x + 8), vld1q_f32(x + 12) };
  return mv16_blk16v(xr, wcol, ldw);
}

static __attribute__((always_inline)) inline
void mv16_blk16(const float *restrict x, const float *restrict wcol,
                int ldw, float *restrict out) {
  vec16 r = mv16_blk16r(x, wcol, ldw);
  vst1q_f32(out, r.v0);      vst1q_f32(out + 4,  r.v1);
  vst1q_f32(out + 8, r.v2);  vst1q_f32(out + 12, r.v3);
}

static __attribute__((always_inline)) inline
float rmsnorm_scale(const float *restrict x) {
  float32x4_t v0 = vld1q_f32(x),     v1 = vld1q_f32(x + 4);
  float32x4_t v2 = vld1q_f32(x + 8), v3 = vld1q_f32(x + 12);
  float32x4_t ss = vmulq_f32(v0, v0);
  ss = vfmaq_f32(ss, v1, v1);
  ss = vfmaq_f32(ss, v2, v2);
  ss = vfmaq_f32(ss, v3, v3);
  return 1.0f / sqrtf(vaddvq_f32(ss) * (1.0f / (float)N_EMBD) + 1e-5f);
}

static __attribute__((always_inline)) inline
void rmsnorm_infer(const float *restrict x, float *restrict out) {
  float32x4_t v0 = vld1q_f32(x),     v1 = vld1q_f32(x + 4);
  float32x4_t v2 = vld1q_f32(x + 8), v3 = vld1q_f32(x + 12);
  float32x4_t ss = vmulq_f32(v0, v0);
  ss = vfmaq_f32(ss, v1, v1);
  ss = vfmaq_f32(ss, v2, v2);
  ss = vfmaq_f32(ss, v3, v3);
  float ms = vaddvq_f32(ss) * (1.0f / (float)N_EMBD);
  float32x4_t vs = vdupq_n_f32(1.0f / sqrtf(ms + 1e-5f));
  vst1q_f32(out,      vmulq_f32(v0, vs));
  vst1q_f32(out + 4,  vmulq_f32(v1, vs));
  vst1q_f32(out + 8,  vmulq_f32(v2, vs));
  vst1q_f32(out + 12, vmulq_f32(v3, vs));
}

#else

typedef struct { __m256 lo, hi; } vec16;

#define ACC(p0, p1, wp, xb)                                          \
  p0 = _mm256_fmadd_ps(_mm256_loadu_ps(wp),       xb, p0);           \
  p1 = _mm256_fmadd_ps(_mm256_loadu_ps((wp) + 8), xb, p1)

#define ACC4(p0, p1, w, ldw, x, c)                                   \
  ACC(p0, p1, (w),               _mm256_broadcast_ss((x) + (c)));     \
  ACC(p0, p1, (w) + (ldw),       _mm256_broadcast_ss((x) + (c) + 1)); \
  ACC(p0, p1, (w) + 2 * (ldw),   _mm256_broadcast_ss((x) + (c) + 2)); \
  ACC(p0, p1, (w) + 3 * (ldw),   _mm256_broadcast_ss((x) + (c) + 3))

static __attribute__((always_inline)) inline
vec16 mv16_blk16r(const float *restrict x, const float *restrict wcol, int ldw) {
  __m256 a0 = _mm256_setzero_ps(), a1 = a0, b0 = a0, b1 = a0;
  const size_t s = (size_t)ldw;
  ACC4(a0, a1, wcol,            s, x, 0);
  ACC4(b0, b1, wcol + 4  * s,   s, x, 4);
  ACC4(a0, a1, wcol + 8  * s,   s, x, 8);
  ACC4(b0, b1, wcol + 12 * s,   s, x, 12);
  vec16 r = { _mm256_add_ps(a0, b0), _mm256_add_ps(a1, b1) };
  return r;
}

static __attribute__((always_inline)) inline
vec16 mv16_blk16r4(const float *restrict x, const float *restrict wcol, int ldw) {
  __m256 a0 = _mm256_setzero_ps(), a1 = a0, b0 = a0, b1 = a0;
  __m256 c0 = a0, c1 = a0, d0 = a0, d1 = a0;
  const size_t s = (size_t)ldw;
  ACC4(a0, a1, wcol,            s, x, 0);
  ACC4(b0, b1, wcol + 4  * s,   s, x, 4);
  ACC4(c0, c1, wcol + 8  * s,   s, x, 8);
  ACC4(d0, d1, wcol + 12 * s,   s, x, 12);
  vec16 r = { _mm256_add_ps(_mm256_add_ps(a0, b0), _mm256_add_ps(c0, d0)),
              _mm256_add_ps(_mm256_add_ps(a1, b1), _mm256_add_ps(c1, d1)) };
  return r;
}

static __attribute__((always_inline)) inline
void mv16_blk16(const float *restrict x, const float *restrict wcol,
                int ldw, float *restrict out) {
  vec16 r = mv16_blk16r(x, wcol, ldw);
  _mm256_storeu_ps(out,     r.lo);
  _mm256_storeu_ps(out + 8, r.hi);
}

static __attribute__((always_inline)) inline
void mv16_blk8(const float *restrict x, const float *restrict wcol,
               int ldw, float *restrict out) {
  __m256 a = _mm256_setzero_ps(), b = a;
  const size_t s = (size_t)ldw;
  for (int c = 0; c < 16; c += 2) {
    a = _mm256_fmadd_ps(_mm256_loadu_ps(wcol + (size_t)c * s),
                        _mm256_broadcast_ss(x + c), a);
    b = _mm256_fmadd_ps(_mm256_loadu_ps(wcol + (size_t)(c + 1) * s),
                        _mm256_broadcast_ss(x + c + 1), b);
  }
  _mm256_storeu_ps(out, _mm256_add_ps(a, b));
}

static __attribute__((always_inline)) inline
void mv16_blk4(const float *restrict x, const float *restrict wcol,
               int ldw, float *restrict out) {
  __m128 a = _mm_setzero_ps(), b = a;
  const size_t s = (size_t)ldw;
  for (int c = 0; c < 16; c += 2) {
    a = _mm_fmadd_ps(_mm_loadu_ps(wcol + (size_t)c * s),
                     _mm_broadcast_ss(x + c), a);
    b = _mm_fmadd_ps(_mm_loadu_ps(wcol + (size_t)(c + 1) * s),
                     _mm_broadcast_ss(x + c + 1), b);
  }
  _mm_storeu_ps(out, _mm_add_ps(a, b));
}

static __attribute__((always_inline)) inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  return hsum128(_mm_add_ps(lo, hi));
}

static __attribute__((always_inline)) inline
float rmsnorm_scale(const float *restrict x) {
  __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8);
  __m256 ss = _mm256_fmadd_ps(v1, v1, _mm256_mul_ps(v0, v0));
  return 1.0f / sqrtf(hsum256(ss) * (1.0f / (float)N_EMBD) + 1e-5f);
}

static __attribute__((always_inline)) inline
void rmsnorm_infer(const float *restrict x, float *restrict out) {
  __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8);
  __m256 ss = _mm256_fmadd_ps(v1, v1, _mm256_mul_ps(v0, v0));
  float ms = hsum256(ss) * (1.0f / (float)N_EMBD);
  __m256 vs = _mm256_set1_ps(1.0f / sqrtf(ms + 1e-5f));
  _mm256_storeu_ps(out,     _mm256_mul_ps(v0, vs));
  _mm256_storeu_ps(out + 8, _mm256_mul_ps(v1, vs));
}

#endif

static void build_pretok(void) {
  size_t n = (size_t)vocab_size * BLOCK_SIZE;
  size_t bytes = (n * sizeof(PreTok) + 63u) & ~(size_t)63u;
  pretok = (PreTok *)MG_ALIGNED_ALLOC(64, bytes);
  if (!pretok) {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }
  for (int t = 0; t < vocab_size; t++)
    for (int p = 0; p < BLOCK_SIZE; p++) {
      float x[N_EMBD], xr[N_EMBD], xn[N_EMBD];
      for (int i = 0; i < N_EMBD; i++)
        x[i] = wte[(size_t)t * N_EMBD + i] + wpe[(size_t)p * N_EMBD + i];
      rmsnorm_infer(x, xr);
      PreTok *e = &pretok[(size_t)t * BLOCK_SIZE + p];
      memcpy(e->xin, xr, sizeof(xr));
      rmsnorm_infer(xr, xn);
      mv16_blk16(xn, iw_q[0], N_EMBD, e->q);
      mv16_blk16(xn, iw_k[0], N_EMBD, e->k);
      mv16_blk16(xn, iw_v[0], N_EMBD, e->v);
    }
}

#ifdef MG_SIMD_NEON

static void gpt_forward_infer(int token_id, int pos_id, float *restrict logits_out) {
  float x[N_EMBD], xn[N_EMBD], xin[N_EMBD];
  float q[N_EMBD];

  for (int li = 0; li < N_LAYER; li++) {
    const float *xin_p;
    if (li == 0) {

      const PreTok *pt = pretok + ((size_t)token_id * BLOCK_SIZE + pos_id);
      xin_p = pt->xin;
      memcpy(q, pt->q, sizeof(q));
      memcpy(kv_keys[li][pos_id], pt->k, N_EMBD * sizeof(float));
      memcpy(kv_vals[li][pos_id], pt->v, N_EMBD * sizeof(float));
    } else {
      memcpy(xin, x, sizeof(x));
      xin_p = xin;
      rmsnorm_infer(x, xn);
      mv16_blk16(xn, iw_q[li], N_EMBD, q);
      mv16_blk16(xn, iw_k[li], N_EMBD, kv_keys[li][pos_id]);
      mv16_blk16(xn, iw_v[li], N_EMBD, kv_vals[li][pos_id]);
    }

    int seq_len = pos_id + 1;
    float32x4_t q0 = vld1q_f32(q),     q1 = vld1q_f32(q + 4);
    float32x4_t q2 = vld1q_f32(q + 8), q3 = vld1q_f32(q + 12);
    float32x4_t sc[BLOCK_SIZE];

#define SCORE(dst, tt)                                                    \
    do {                                                                  \
      const float *kt_ = kv_keys[li][tt];                                 \
      float32x4_t y0 = vmulq_f32(q0, vld1q_f32(kt_));                     \
      float32x4_t y1 = vmulq_f32(q1, vld1q_f32(kt_ + 4));                 \
      float32x4_t y2 = vmulq_f32(q2, vld1q_f32(kt_ + 8));                 \
      float32x4_t y3 = vmulq_f32(q3, vld1q_f32(kt_ + 12));                \
      dst = vmulq_n_f32(vpaddq_f32(vpaddq_f32(y0, y1),                    \
                                   vpaddq_f32(y2, y3)), ATTN_SCALE);      \
    } while (0)

    float32x4_t m0 = vdupq_n_f32(-3.0e38f), m1 = m0;
    int tt = 0;
    for (; tt + 1 < seq_len; tt += 2) {
      SCORE(sc[tt],     tt);
      SCORE(sc[tt + 1], tt + 1);
      m0 = vmaxq_f32(m0, sc[tt]);
      m1 = vmaxq_f32(m1, sc[tt + 1]);
    }
    if (tt < seq_len) { SCORE(sc[tt], tt); m0 = vmaxq_f32(m0, sc[tt]); }
    float32x4_t mx = vmaxq_f32(m0, m1);
#undef SCORE

    float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
    float32x4_t o0 = s0, o1 = s0, o2 = s0, o3 = s0;
    float32x4_t u0 = s0, u1 = s0, u2 = s0, u3 = s0;

#define ACCUM_V(e, tt, r0, r1, r2, r3)                                    \
    do {                                                                  \
      const float *vt_ = kv_vals[li][tt];                                 \
      r0 = vfmaq_laneq_f32(r0, vld1q_f32(vt_),      e, 0);                \
      r1 = vfmaq_laneq_f32(r1, vld1q_f32(vt_ + 4),  e, 1);                \
      r2 = vfmaq_laneq_f32(r2, vld1q_f32(vt_ + 8),  e, 2);                \
      r3 = vfmaq_laneq_f32(r3, vld1q_f32(vt_ + 12), e, 3);                \
    } while (0)

    tt = 0;
    for (; tt + 1 < seq_len; tt += 2) {
      float32x4_t e0 = vfexpq(vsubq_f32(sc[tt],     mx));
      float32x4_t e1 = vfexpq(vsubq_f32(sc[tt + 1], mx));
      s0 = vaddq_f32(s0, e0);
      s1 = vaddq_f32(s1, e1);
      ACCUM_V(e0, tt,     o0, o1, o2, o3);
      ACCUM_V(e1, tt + 1, u0, u1, u2, u3);
    }
    if (tt < seq_len) {
      float32x4_t e0 = vfexpq(vsubq_f32(sc[tt], mx));
      s0 = vaddq_f32(s0, e0);
      ACCUM_V(e0, tt, o0, o1, o2, o3);
    }
#undef ACCUM_V

    float32x4_t inv = vdivq_f32(vdupq_n_f32(1.0f), vaddq_f32(s0, s1));

    vec16 ao = { vmulq_laneq_f32(vaddq_f32(o0, u0), inv, 0),
                 vmulq_laneq_f32(vaddq_f32(o1, u1), inv, 1),
                 vmulq_laneq_f32(vaddq_f32(o2, u2), inv, 2),
                 vmulq_laneq_f32(vaddq_f32(o3, u3), inv, 3) };

    vec16 wo = mv16_blk16v(ao, iw_o[li], N_EMBD);
    vst1q_f32(x,      vaddq_f32(wo.v0, vld1q_f32(xin_p)));
    vst1q_f32(x + 4,  vaddq_f32(wo.v1, vld1q_f32(xin_p + 4)));
    vst1q_f32(x + 8,  vaddq_f32(wo.v2, vld1q_f32(xin_p + 8)));
    vst1q_f32(x + 12, vaddq_f32(wo.v3, vld1q_f32(xin_p + 12)));

    const float s_mlp = rmsnorm_scale(x);
    const float32x4_t z = vdupq_n_f32(0);
    float h2[MLP_DIM];
    for (int b = 0; b < MLP_DIM; b += 16) {
      vec16 h = mv16_blk16r4(x, iw_fc1[li] + b * 16, N_EMBD);
      float32x4_t r0 = vmaxq_f32(h.v0, z), r1 = vmaxq_f32(h.v1, z);
      float32x4_t r2 = vmaxq_f32(h.v2, z), r3 = vmaxq_f32(h.v3, z);
      vst1q_f32(h2 + b,      vmulq_f32(r0, r0));
      vst1q_f32(h2 + b + 4,  vmulq_f32(r1, r1));
      vst1q_f32(h2 + b + 8,  vmulq_f32(r2, r2));
      vst1q_f32(h2 + b + 12, vmulq_f32(r3, r3));
    }
    float32x4_t t0 = z, t1 = z, t2 = z, t3 = z;
    float32x4_t n0 = z, n1 = z, n2 = z, n3 = z;
    float32x4_t y0 = z, y1 = z, y2 = z, y3 = z;
    float32x4_t w0 = z, w1 = z, w2 = z, w3 = z;
    for (int c = 0; c < MLP_DIM; c += 16) {
      const float *wf = iw_fc2[li] + (size_t)c * N_EMBD;
      ACC4(t0, t1, t2, t3, wf,                       (size_t)N_EMBD, vld1q_f32(h2 + c));
      ACC4(n0, n1, n2, n3, wf + 4  * (size_t)N_EMBD, (size_t)N_EMBD, vld1q_f32(h2 + c + 4));
      ACC4(y0, y1, y2, y3, wf + 8  * (size_t)N_EMBD, (size_t)N_EMBD, vld1q_f32(h2 + c + 8));
      ACC4(w0, w1, w2, w3, wf + 12 * (size_t)N_EMBD, (size_t)N_EMBD, vld1q_f32(h2 + c + 12));
    }

    const float32x4_t sv = vdupq_n_f32(s_mlp * s_mlp);
    vst1q_f32(x,      vfmaq_f32(vld1q_f32(x),      vaddq_f32(vaddq_f32(t0, n0), vaddq_f32(y0, w0)), sv));
    vst1q_f32(x + 4,  vfmaq_f32(vld1q_f32(x + 4),  vaddq_f32(vaddq_f32(t1, n1), vaddq_f32(y1, w1)), sv));
    vst1q_f32(x + 8,  vfmaq_f32(vld1q_f32(x + 8),  vaddq_f32(vaddq_f32(t2, n2), vaddq_f32(y2, w2)), sv));
    vst1q_f32(x + 12, vfmaq_f32(vld1q_f32(x + 12), vaddq_f32(vaddq_f32(t3, n3), vaddq_f32(y3, w3)), sv));
  }

  int r0 = 0;
  for (; r0 + 16 <= lm_pad; r0 += 16)
    mv16_blk16(x, iw_lm + r0, lm_pad, logits_out + r0);
  if (r0 + 8 <= lm_pad) { mv16_blk8(x, iw_lm + r0, lm_pad, logits_out + r0); r0 += 8; }
  if (r0 + 4 <= lm_pad) { mv16_blk4(x, iw_lm + r0, lm_pad, logits_out + r0); r0 += 4; }
}

#else

static void gpt_forward_infer(int token_id, int pos_id, float *restrict logits_out) {
  float x[N_EMBD], xn[N_EMBD], xin[N_EMBD];
  float q[N_EMBD];

  for (int li = 0; li < N_LAYER; li++) {
    const float *xin_p;
    if (li == 0) {
      const PreTok *pt = pretok + ((size_t)token_id * BLOCK_SIZE + pos_id);
      xin_p = pt->xin;
      memcpy(q, pt->q, sizeof(q));
      memcpy(kv_keys[li][pos_id], pt->k, N_EMBD * sizeof(float));
      memcpy(kv_vals[li][pos_id], pt->v, N_EMBD * sizeof(float));
    } else {
      memcpy(xin, x, sizeof(x));
      xin_p = xin;
      rmsnorm_infer(x, xn);
      mv16_blk16(xn, iw_q[li], N_EMBD, q);
      mv16_blk16(xn, iw_k[li], N_EMBD, kv_keys[li][pos_id]);
      mv16_blk16(xn, iw_v[li], N_EMBD, kv_vals[li][pos_id]);
    }

    int seq_len = pos_id + 1;
    __m128 q0 = _mm_loadu_ps(q),     q1 = _mm_loadu_ps(q + 4);
    __m128 q2 = _mm_loadu_ps(q + 8), q3 = _mm_loadu_ps(q + 12);
    __m128 sc[BLOCK_SIZE];

#define SCORE(dst, tt)                                                    \
    do {                                                                  \
      const float *kt_ = kv_keys[li][tt];                                 \
      __m128 y0 = _mm_mul_ps(q0, _mm_loadu_ps(kt_));                      \
      __m128 y1 = _mm_mul_ps(q1, _mm_loadu_ps(kt_ + 4));                  \
      __m128 y2 = _mm_mul_ps(q2, _mm_loadu_ps(kt_ + 8));                  \
      __m128 y3 = _mm_mul_ps(q3, _mm_loadu_ps(kt_ + 12));                 \
      dst = _mm_mul_ps(_mm_hadd_ps(_mm_hadd_ps(y0, y1),                   \
                                   _mm_hadd_ps(y2, y3)),                  \
                       _mm_set1_ps(ATTN_SCALE));                          \
    } while (0)

    __m128 m0 = _mm_set1_ps(-3.0e38f), m1 = m0;
    int tt = 0;
    for (; tt + 1 < seq_len; tt += 2) {
      SCORE(sc[tt],     tt);
      SCORE(sc[tt + 1], tt + 1);
      m0 = _mm_max_ps(m0, sc[tt]);
      m1 = _mm_max_ps(m1, sc[tt + 1]);
    }
    if (tt < seq_len) { SCORE(sc[tt], tt); m0 = _mm_max_ps(m0, sc[tt]); }
    __m128 mx = _mm_max_ps(m0, m1);
#undef SCORE

    __m128 s0 = _mm_setzero_ps(), s1 = s0;
    __m128 o0 = s0, o1 = s0, o2 = s0, o3 = s0;
    __m128 u0 = s0, u1 = s0, u2 = s0, u3 = s0;

#define ACCUM_V(e, tt, r0, r1, r2, r3)                                    \
    do {                                                                  \
      const float *vt_ = kv_vals[li][tt];                                 \
      r0 = _mm_fmadd_ps(_mm_loadu_ps(vt_),                                \
                        _mm_shuffle_ps(e, e, _MM_SHUFFLE(0,0,0,0)), r0);  \
      r1 = _mm_fmadd_ps(_mm_loadu_ps(vt_ + 4),                            \
                        _mm_shuffle_ps(e, e, _MM_SHUFFLE(1,1,1,1)), r1);  \
      r2 = _mm_fmadd_ps(_mm_loadu_ps(vt_ + 8),                            \
                        _mm_shuffle_ps(e, e, _MM_SHUFFLE(2,2,2,2)), r2);  \
      r3 = _mm_fmadd_ps(_mm_loadu_ps(vt_ + 12),                           \
                        _mm_shuffle_ps(e, e, _MM_SHUFFLE(3,3,3,3)), r3);  \
    } while (0)

    tt = 0;
    for (; tt + 1 < seq_len; tt += 2) {
      __m128 e0 = vfexpq(_mm_sub_ps(sc[tt],     mx));
      __m128 e1 = vfexpq(_mm_sub_ps(sc[tt + 1], mx));
      s0 = _mm_add_ps(s0, e0);
      s1 = _mm_add_ps(s1, e1);
      ACCUM_V(e0, tt,     o0, o1, o2, o3);
      ACCUM_V(e1, tt + 1, u0, u1, u2, u3);
    }
    if (tt < seq_len) {
      __m128 e0 = vfexpq(_mm_sub_ps(sc[tt], mx));
      s0 = _mm_add_ps(s0, e0);
      ACCUM_V(e0, tt, o0, o1, o2, o3);
    }
#undef ACCUM_V

    __m128 inv = _mm_div_ps(_mm_set1_ps(1.0f), _mm_add_ps(s0, s1));
    float ao[N_EMBD];
    _mm_storeu_ps(ao,      _mm_mul_ps(_mm_add_ps(o0, u0),
                           _mm_shuffle_ps(inv, inv, _MM_SHUFFLE(0,0,0,0))));
    _mm_storeu_ps(ao + 4,  _mm_mul_ps(_mm_add_ps(o1, u1),
                           _mm_shuffle_ps(inv, inv, _MM_SHUFFLE(1,1,1,1))));
    _mm_storeu_ps(ao + 8,  _mm_mul_ps(_mm_add_ps(o2, u2),
                           _mm_shuffle_ps(inv, inv, _MM_SHUFFLE(2,2,2,2))));
    _mm_storeu_ps(ao + 12, _mm_mul_ps(_mm_add_ps(o3, u3),
                           _mm_shuffle_ps(inv, inv, _MM_SHUFFLE(3,3,3,3))));

    vec16 wo = mv16_blk16r(ao, iw_o[li], N_EMBD);
    _mm256_storeu_ps(x,     _mm256_add_ps(wo.lo, _mm256_loadu_ps(xin_p)));
    _mm256_storeu_ps(x + 8, _mm256_add_ps(wo.hi, _mm256_loadu_ps(xin_p + 8)));

    const float s_mlp = rmsnorm_scale(x);
    const __m256 z = _mm256_setzero_ps();
    float h2[MLP_DIM];
    for (int b = 0; b < MLP_DIM; b += 16) {
      vec16 h = mv16_blk16r4(x, iw_fc1[li] + b * 16, N_EMBD);
      __m256 r0 = _mm256_max_ps(h.lo, z), r1 = _mm256_max_ps(h.hi, z);
      _mm256_storeu_ps(h2 + b,     _mm256_mul_ps(r0, r0));
      _mm256_storeu_ps(h2 + b + 8, _mm256_mul_ps(r1, r1));
    }
    __m256 t0 = z, t1 = z, n0 = z, n1 = z, y0 = z, y1 = z, w0 = z, w1 = z;
    for (int c = 0; c < MLP_DIM; c += 16) {
      const float *wf = iw_fc2[li] + (size_t)c * N_EMBD;
      ACC4(t0, t1, wf,                       (size_t)N_EMBD, h2, c);
      ACC4(n0, n1, wf + 4  * (size_t)N_EMBD, (size_t)N_EMBD, h2, c + 4);
      ACC4(y0, y1, wf + 8  * (size_t)N_EMBD, (size_t)N_EMBD, h2, c + 8);
      ACC4(w0, w1, wf + 12 * (size_t)N_EMBD, (size_t)N_EMBD, h2, c + 12);
    }
    const __m256 sv = _mm256_set1_ps(s_mlp * s_mlp);
    __m256 S0 = _mm256_add_ps(_mm256_add_ps(t0, n0), _mm256_add_ps(y0, w0));
    __m256 S1 = _mm256_add_ps(_mm256_add_ps(t1, n1), _mm256_add_ps(y1, w1));
    _mm256_storeu_ps(x,     _mm256_fmadd_ps(S0, sv, _mm256_loadu_ps(x)));
    _mm256_storeu_ps(x + 8, _mm256_fmadd_ps(S1, sv, _mm256_loadu_ps(x + 8)));
  }

  int r0 = 0;
  for (; r0 + 16 <= lm_pad; r0 += 16)
    mv16_blk16(x, iw_lm + r0, lm_pad, logits_out + r0);
  if (r0 + 8 <= lm_pad) { mv16_blk8(x, iw_lm + r0, lm_pad, logits_out + r0); r0 += 8; }
  if (r0 + 4 <= lm_pad) { mv16_blk4(x, iw_lm + r0, lm_pad, logits_out + r0); r0 += 4; }
}

#endif

static void gpt_backward(int n, const int *tokens, const int *targets) {
  memset(dk_accum, 0, sizeof(dk_accum));
  memset(dv_accum, 0, sizeof(dv_accum));
  float inv_n = 1.0f / n;

  for (int pos = n - 1; pos >= 0; pos--) {
    PosActs *act = &saved[pos];
    int seq_len = pos + 1;

    float dl[MAX_CHARS + 1];
    for (int i = 0; i < vocab_size; i++)
      dl[i] = (saved_probs[pos][i] - (i == targets[pos] ? 1.0f : 0.0f)) * inv_n;

    float dx[N_EMBD];
    memset(dx, 0, sizeof(dx));
    linear_bwd_x(lm_head, dl, vocab_size, N_EMBD, dx);
    linear_bwd_w(act->x_out, dl, vocab_size, N_EMBD, d_lm_head);

    for (int li = N_LAYER - 1; li >= 0; li--) {

      float d_h2[MLP_DIM];
      memset(d_h2, 0, sizeof(d_h2));
      linear_bwd_x(mlp_fc2[li], dx, N_EMBD, MLP_DIM, d_h2);
      linear_bwd_w(act->mlp_post[li], dx, N_EMBD, MLP_DIM, d_mlp_fc2[li]);

      float d_h1[MLP_DIM];
      for (int i = 0; i < MLP_DIM; i++)
        d_h1[i] =
            act->mlp_pre[li][i] > 0 ? 2.0f * act->mlp_pre[li][i] * d_h2[i] : 0;

      float d_xn_mlp[N_EMBD];
      memset(d_xn_mlp, 0, sizeof(d_xn_mlp));
      linear_bwd_x(mlp_fc1[li], d_h1, MLP_DIM, N_EMBD, d_xn_mlp);
      linear_bwd_w(act->xn_mlp[li], d_h1, MLP_DIM, N_EMBD, d_mlp_fc1[li]);

      float d_x_mid[N_EMBD];
      memset(d_x_mid, 0, sizeof(d_x_mid));
      rmsnorm_bwd(act->x_mid[li], act->rms_scale_mlp[li], d_xn_mlp, N_EMBD,
                  d_x_mid);
      for (int i = 0; i < N_EMBD; i++)
        dx[i] += d_x_mid[i];

      float d_ao[N_EMBD];
      memset(d_ao, 0, sizeof(d_ao));
      linear_bwd_x(attn_wo[li], dx, N_EMBD, N_EMBD, d_ao);
      linear_bwd_w(act->attn_out[li], dx, N_EMBD, N_EMBD, d_attn_wo[li]);

      float d_q[N_EMBD];
      memset(d_q, 0, sizeof(d_q));
      float scale = 1.0f / sqrtf((float)N_EMBD / (float)N_HEAD);

      for (int h = 0; h < N_HEAD; h++) {
        int hs = h * HEAD_DIM;
        float d_aw[BLOCK_SIZE];
        memset(d_aw, 0, sizeof(d_aw));
        for (int j = 0; j < HEAD_DIM; j++) {
          for (int tt = 0; tt < seq_len; tt++) {
            d_aw[tt] += d_ao[hs + j] * kv_vals[li][tt][hs + j];
            dv_accum[li][tt][hs + j] += act->aw[li][h][tt] * d_ao[hs + j];
          }
        }
        float dot = 0;
        for (int tt = 0; tt < seq_len; tt++)
          dot += d_aw[tt] * act->aw[li][h][tt];
        float d_al[BLOCK_SIZE];
        for (int tt = 0; tt < seq_len; tt++)
          d_al[tt] = act->aw[li][h][tt] * (d_aw[tt] - dot);
        for (int tt = 0; tt < seq_len; tt++) {
          for (int j = 0; j < HEAD_DIM; j++) {
            d_q[hs + j] += d_al[tt] * kv_keys[li][tt][hs + j] * scale;
            dk_accum[li][tt][hs + j] += d_al[tt] * act->q[li][hs + j] * scale;
          }
        }
      }

      float d_xn[N_EMBD];
      memset(d_xn, 0, sizeof(d_xn));
      linear_bwd_x(attn_wq[li], d_q, N_EMBD, N_EMBD, d_xn);
      linear_bwd_w(act->xn_attn[li], d_q, N_EMBD, N_EMBD, d_attn_wq[li]);
      linear_bwd_x(attn_wk[li], dk_accum[li][pos], N_EMBD, N_EMBD, d_xn);
      linear_bwd_w(act->xn_attn[li], dk_accum[li][pos], N_EMBD, N_EMBD,
                   d_attn_wk[li]);
      linear_bwd_x(attn_wv[li], dv_accum[li][pos], N_EMBD, N_EMBD, d_xn);
      linear_bwd_w(act->xn_attn[li], dv_accum[li][pos], N_EMBD, N_EMBD,
                   d_attn_wv[li]);

      float d_x_in[N_EMBD];
      memset(d_x_in, 0, sizeof(d_x_in));
      rmsnorm_bwd(act->x_in[li], act->rms_scale_attn[li], d_xn, N_EMBD, d_x_in);
      for (int i = 0; i < N_EMBD; i++)
        dx[i] = dx[i] + d_x_in[i];
    }

    float d_embed[N_EMBD];
    memset(d_embed, 0, sizeof(d_embed));
    rmsnorm_bwd(act->x_embed, act->rms_scale_init, dx, N_EMBD, d_embed);

    int tok = tokens[pos];
    for (int i = 0; i < N_EMBD; i++) {
      d_wte[tok * N_EMBD + i] += d_embed[i];
      d_wpe[pos * N_EMBD + i] += d_embed[i];
    }
  }
}

static void adam_update(float *p, float *g, float *m, float *v, int sz,
                        float lr, float b1, float b2, float eps, int step) {
  float b1c = 1.0f - powf(b1, step + 1);
  float b2c = 1.0f - powf(b2, step + 1);
  for (int i = 0; i < sz; i++) {
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    p[i] -= lr * (m[i] / b1c) / (sqrtf(v[i] / b2c) + eps);
    g[i] = 0;
  }
}

int main(int argc, char **argv) {
  load_dataset(argc > 1 ? argv[1] : "data/names.txt");

  int *doc_order = (int *)malloc(num_docs * sizeof(int));
  for (int i = 0; i < num_docs; i++)
    doc_order[i] = i;
  shuffle_ints(doc_order, num_docs);
  char (*docs_tmp)[MAX_DOC_LEN] = malloc((size_t)num_docs * MAX_DOC_LEN);
  for (int i = 0; i < num_docs; i++)
    memcpy(docs_tmp[i], docs[doc_order[i]], MAX_DOC_LEN);
  memcpy(docs, docs_tmp, (size_t)num_docs * MAX_DOC_LEN);
  free(docs_tmp);
  free(doc_order);

  printf("num docs: %d\n", num_docs);
  build_tokenizer();
  printf("vocab size: %d\n", vocab_size);
  init_params();

  float lr = 3e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  float running_loss = 3.3f;
  int num_steps = 20000;

  for (int step = 0; step < num_steps; step++) {
    char *doc = docs[step % num_docs];
    int doc_len = (int)strlen(doc);

    int tokens[MAX_DOC_LEN + 2], targets[BLOCK_SIZE];
    tokens[0] = BOS;
    for (int i = 0; i < doc_len; i++)
      tokens[i + 1] = char_to_id(doc[i]);
    tokens[doc_len + 1] = BOS;
    int n = BLOCK_SIZE < (doc_len + 1) ? BLOCK_SIZE : (doc_len + 1);

    memset(kv_keys, 0, sizeof(kv_keys));
    memset(kv_vals, 0, sizeof(kv_vals));

    float total_loss = 0;
    float logits[MAX_CHARS + 1];
    for (int pos = 0; pos < n; pos++) {
      targets[pos] = tokens[pos + 1];
      gpt_forward(tokens[pos], pos, logits, &saved[pos]);
      softmax_fwd_precise(logits, vocab_size, saved_probs[pos]);
      total_loss += -logf(saved_probs[pos][targets[pos]] + 1e-30f);
    }
    float loss = total_loss / n;

    gpt_backward(n, tokens, targets);

    {
      float gnorm2 = 0;
      int es2 = vocab_size * N_EMBD, ps2 = BLOCK_SIZE * N_EMBD;
      int as2 = N_EMBD * N_EMBD, ms2 = MLP_DIM * N_EMBD;
      for (int i = 0; i < es2; i++) gnorm2 += d_wte[i]*d_wte[i] + d_lm_head[i]*d_lm_head[i];
      for (int i = 0; i < ps2; i++) gnorm2 += d_wpe[i]*d_wpe[i];
      for (int l = 0; l < N_LAYER; l++) {
        for (int i = 0; i < as2; i++)
          gnorm2 += d_attn_wq[l][i]*d_attn_wq[l][i] + d_attn_wk[l][i]*d_attn_wk[l][i]
                  + d_attn_wv[l][i]*d_attn_wv[l][i] + d_attn_wo[l][i]*d_attn_wo[l][i];
        for (int i = 0; i < ms2; i++)
          gnorm2 += d_mlp_fc1[l][i]*d_mlp_fc1[l][i] + d_mlp_fc2[l][i]*d_mlp_fc2[l][i];
      }
      float gnorm = sqrtf(gnorm2);
      float clip = 1.0f;
      if (gnorm > clip) {
        float scale = clip / gnorm;
        for (int i = 0; i < es2; i++) { d_wte[i]*=scale; d_lm_head[i]*=scale; }
        for (int i = 0; i < ps2; i++) d_wpe[i]*=scale;
        for (int l = 0; l < N_LAYER; l++) {
          for (int i = 0; i < as2; i++) {
            d_attn_wq[l][i]*=scale; d_attn_wk[l][i]*=scale;
            d_attn_wv[l][i]*=scale; d_attn_wo[l][i]*=scale;
          }
          for (int i = 0; i < ms2; i++) { d_mlp_fc1[l][i]*=scale; d_mlp_fc2[l][i]*=scale; }
        }
      }
    }
    float lr_t =
        lr * 0.5f * (1.0f + cosf((float)M_PI * step / (float)num_steps));
    int es = vocab_size * N_EMBD, ps = BLOCK_SIZE * N_EMBD;
    int as = N_EMBD * N_EMBD, ms = MLP_DIM * N_EMBD;
    adam_update(wte, d_wte, adam_m_wte, adam_v_wte, es, lr_t, b1, b2, eps,
                step);
    adam_update(wpe, d_wpe, adam_m_wpe, adam_v_wpe, ps, lr_t, b1, b2, eps,
                step);
    adam_update(lm_head, d_lm_head, adam_m_lm, adam_v_lm, es, lr_t, b1, b2, eps,
                step);
    for (int i = 0; i < N_LAYER; i++) {
      adam_update(attn_wq[i], d_attn_wq[i], adam_m_wq[i], adam_v_wq[i], as,
                  lr_t, b1, b2, eps, step);
      adam_update(attn_wk[i], d_attn_wk[i], adam_m_wk[i], adam_v_wk[i], as,
                  lr_t, b1, b2, eps, step);
      adam_update(attn_wv[i], d_attn_wv[i], adam_m_wv[i], adam_v_wv[i], as,
                  lr_t, b1, b2, eps, step);
      adam_update(attn_wo[i], d_attn_wo[i], adam_m_wo[i], adam_v_wo[i], as,
                  lr_t, b1, b2, eps, step);
      adam_update(mlp_fc1[i], d_mlp_fc1[i], adam_m_fc1[i], adam_v_fc1[i], ms,
                  lr_t, b1, b2, eps, step);
      adam_update(mlp_fc2[i], d_mlp_fc2[i], adam_m_fc2[i], adam_v_fc2[i], ms,
                  lr_t, b1, b2, eps, step);
    }

    running_loss = running_loss * 0.99f + loss * 0.01f;
    if ((step + 1) % 100 == 0 || step == 0 || step == num_steps - 1)
      printf("step %4d / %4d | loss %.4f  (avg %.4f)\n",
             step + 1, num_steps, loss, running_loss);
  }

  float temperature = 0.5f;
  printf("\ninference\n");

  infer_pack_weights();
  const float inv_t = 1.0f / temperature;

  for (int si = 0; si < 10; si++) {
    int token_id = BOS;
    char buf[BLOCK_SIZE + 1] = {0};
    int len = 0;
    for (int pos = 0; pos < BLOCK_SIZE; pos++) {
      float logits[LM_PAD_MAX];
      gpt_forward_infer(token_id, pos, logits);
      token_id = sample_logits(logits, vocab_size, lm_pad, inv_t);
      if (token_id == BOS) break;
      if (token_id < num_uchars) buf[len++] = uchars_arr[token_id];
    }
    printf("sample %2d: %s\n", si + 1, buf);
    memset(kv_keys, 0, sizeof(kv_keys));
    memset(kv_vals, 0, sizeof(kv_vals));
  }

  long long N = 5000000LL;
  long long emitted = 0;
  int tok = BOS, pos = 0;
  double t0 = now_s();

  float logits[LM_PAD_MAX];

  for (long long i = 0; i < N; i++) {
    if (pos >= BLOCK_SIZE) { pos = 0; }
    gpt_forward_infer(tok, pos, logits);
    int nxt = sample_logits(logits, vocab_size, lm_pad, inv_t);
    if (nxt == BOS) { tok = BOS; pos = 0; }
    else { tok = nxt; pos++; }
    emitted++;
  }

  double elapsed = now_s() - t0;
  printf("  c fp32+%s %14.0f tok/sec\n", MG_SIMD_NAME, emitted / elapsed);

  free(wte);
  free(d_wte);
  free(adam_m_wte);
  free(adam_v_wte);
  free(wpe);
  free(d_wpe);
  free(adam_m_wpe);
  free(adam_v_wpe);
  free(lm_head);
  free(d_lm_head);
  free(adam_m_lm);
  free(adam_v_lm);
  for (int i = 0; i < N_LAYER; i++) {
    free(attn_wq[i]);
    free(d_attn_wq[i]);
    free(adam_m_wq[i]);
    free(adam_v_wq[i]);
    free(attn_wk[i]);
    free(d_attn_wk[i]);
    free(adam_m_wk[i]);
    free(adam_v_wk[i]);
    free(attn_wv[i]);
    free(d_attn_wv[i]);
    free(adam_m_wv[i]);
    free(adam_v_wv[i]);
    free(attn_wo[i]);
    free(d_attn_wo[i]);
    free(adam_m_wo[i]);
    free(adam_v_wo[i]);
    free(mlp_fc1[i]);
    free(d_mlp_fc1[i]);
    free(adam_m_fc1[i]);
    free(adam_v_fc1[i]);
    free(mlp_fc2[i]);
    free(d_mlp_fc2[i]);
    free(adam_m_fc2[i]);
    free(adam_v_fc2[i]);
  }
  MG_ALIGNED_FREE(pretok);
  return 0;
}
