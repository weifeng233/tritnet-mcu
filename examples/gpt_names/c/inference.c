/*
 * BitNet-GPT Inference Engine
 * Ternary weight char-level GPT for MCU deployment
 *
 * Architecture: 1-layer transformer, 4 heads, 16-dim, 64-dim MLP
 * Weights: Ternary {-1, 0, +1} with base-3 packing (10 trits per uint16)
 * Activations: float32 (compact model, float is fine for MCU with FPU)
 *   For MCU without FPU, replace with int8+shift (see ShiftNorm pattern)
 *
 * Key optimization: Ternary MVM uses only adds/subtracts, no multiplication!
 *
 * Based on:
 *   - microgpt-c (pure C GPT, AVX2/FMA)
 *   - BitNetMCU (ternary quantization for MCU)
 *
 * Compile (x86 test): gcc -O3 -o bitnet_gpt.exe bitnet_gpt_inference.c -lm
 * Compile (MCU):       Use your toolchain's float support
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "model.h"
#include "inference.h"

/* ====================================================================
 * RNG (minimal xorshift, same as microgpt-c)
 * ==================================================================== */

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float rng_uniform(uint32_t *state) {
    return (float)(xorshift32(state) >> 8) / 16777216.0f;
}

/* ====================================================================
 * Ternary Weight Unpacking (Base-3 Decoding)
 *
 * Packed format: 10 trits per uint16 word
 *   Encoding: +1->0, -1->1, 0->2
 *   Packed: (base3_value * 65536 + 59048) / 59049
 *
 * Unpacking: reverse the scaling, extract trits via mod 3
 * ==================================================================== */

static void unpack_ternary(const uint16_t *packed, int8_t *out, int n_out) {
    int n_packed = (n_out + 9) / 10;

    for (int i = 0; i < n_packed; i++) {
        uint32_t chunk = packed[i];

        /* BitNetMCU's multiply-by-3 unpacking method.
         * Multiply by 3 to shift each trit into bits 16-17.
         * bit 17 = 1 means trit is 0 (zero weight)
         * bit 16 = 1 means trit is 1 (negative weight)
         * Otherwise trit is 0 (positive weight) */
        for (int t = 0; t < 10; t++) {
            int idx = i * 10 + t;
            if (idx >= n_out) return;

            chunk += (chunk << 1);  /* chunk *= 3 using shift+add */
            if (!(chunk & 0x20000)) {       /* bit 17 = 0: non-zero */
                out[idx] = (chunk & 0x10000) ? -1 : 1;
            } else {
                out[idx] = 0;
            }
            chunk &= 0xFFFF;  /* Clear upper bits for next iteration */
        }
    }
}

/* ====================================================================
 * Ternary Matrix-Vector Multiply
 *
 * y[i] = scale * sum_k( w_ternary[k] * x[k] )
 *
 * Since w is {-1, 0, +1}, inner loop is just adds/subtracts!
 * This is the key performance advantage of ternary weights.
 * ==================================================================== */

static void ternary_mvm(const uint16_t *packed_weights,
                        const float *x, float *y,
                        int n_in, int n_out) {
    int packed_per_row = (n_in + 9) / 10;

    /* Per-token activation quantization (matches BitLinear.activation_quant).
     * Scale input to int8 range [-128, 127] to match QAT training. */
    float max_abs = 0.0f;
    for (int k = 0; k < n_in; k++) {
        float a = x[k] >= 0 ? x[k] : -x[k];
        if (a > max_abs) max_abs = a;
    }
    float input_scale = 127.0f / (max_abs > 1e-5f ? max_abs : 1e-5f);

    /* Quantize input to int8 range */
    int8_t x_q[BITNET_N_EMBD > BITNET_MLP_DIM ? BITNET_N_EMBD : BITNET_MLP_DIM];
    for (int k = 0; k < n_in; k++) {
        int val = (int)(x[k] * input_scale + (x[k] >= 0 ? 0.5f : -0.5f));
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        x_q[k] = (int8_t)val;
    }

    /* Ternary MVM with quantized input (no multiplications!)
     * Ternary weights {-1,0,+1} with int8 input → just adds/subtracts */
    for (int i = 0; i < n_out; i++) {
        int8_t w_row[BITNET_N_EMBD > BITNET_MLP_DIM ? BITNET_N_EMBD : BITNET_MLP_DIM];
        unpack_ternary(packed_weights + i * packed_per_row, w_row, n_in);

        int32_t sum = 0;
        for (int k = 0; k < n_in; k++) {
            int8_t w = w_row[k];
            if (w > 0)       sum += x_q[k];
            else if (w < 0)  sum -= x_q[k];
        }
        /* Dequantize: divide by input_scale to undo the activation scaling.
         * In QAT: y = x_q @ w_q^T / (x_scale * w_scale) ≈ x_q @ w_q^T / x_scale
         * since w_scale ≈ 1 for ternary (weights are already {-1,0,+1}) */
        y[i] = (float)sum / input_scale;
    }
}

/* ====================================================================
 * RMSNorm
 * ==================================================================== */

static void rmsnorm(const float *x, float *out, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / n + 1e-5f);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < n; i++) out[i] = x[i] * inv_rms;
}

/* ====================================================================
 * Softmax
 * ==================================================================== */

static void softmax(const float *x, float *out, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(x[i] - mx);
        sum += out[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) out[i] *= inv;
}

/* ====================================================================
 * Context Init / Reset
 * ==================================================================== */

void bitnet_init(bitnet_ctx_t *ctx) {
    memset(ctx, 0, sizeof(bitnet_ctx_t));
    ctx->rng_state = 42;
    ctx->pos = 0;
}

void bitnet_reset(bitnet_ctx_t *ctx) {
    memset(ctx->kv_keys, 0, sizeof(ctx->kv_keys));
    memset(ctx->kv_vals, 0, sizeof(ctx->kv_vals));
    ctx->pos = 0;
}

/* ====================================================================
 * Forward Pass (one token)
 *
 * Mirrors the Python forward() but uses ternary MVM.
 * All intermediate activations are float32.
 * ==================================================================== */

void bitnet_forward(bitnet_ctx_t *ctx, int token_id, float *logits_out) {
    float x[BITNET_N_EMBD];
    float xn[BITNET_N_EMBD];

    int pos = ctx->pos;
    int seq_len = pos + 1;
    if (seq_len > BITNET_BLOCK_SIZE) seq_len = BITNET_BLOCK_SIZE;
    if (pos >= BITNET_BLOCK_SIZE) pos = BITNET_BLOCK_SIZE - 1;

    /* ---- Token + Position Embedding ---- */
    for (int i = 0; i < BITNET_N_EMBD; i++) {
        x[i] = bitnet_wte[token_id][i] + bitnet_wpe[pos][i];
    }

    /* ---- Initial RMSNorm ---- */
    rmsnorm(x, x, BITNET_N_EMBD);

    /* ---- Transformer Block (single layer) ---- */

    /* -- Attention -- */
    /* Save input for residual */
    float x_in[BITNET_N_EMBD];
    memcpy(x_in, x, sizeof(x_in));

    /* RMSNorm before attention */
    rmsnorm(x, xn, BITNET_N_EMBD);

    /* Q, K, V projections (ternary MVM) */
    float q[BITNET_N_EMBD], k[BITNET_N_EMBD], v[BITNET_N_EMBD];
    ternary_mvm(block0_wq_weights, xn, q,
                     BITNET_N_EMBD, BITNET_N_EMBD);
    ternary_mvm(block0_wk_weights, xn, k,
                     BITNET_N_EMBD, BITNET_N_EMBD);
    ternary_mvm(block0_wv_weights, xn, v,
                     BITNET_N_EMBD, BITNET_N_EMBD);

    /* Store K, V in cache */
    memcpy(ctx->kv_keys[pos], k, sizeof(k));
    memcpy(ctx->kv_vals[pos], v, sizeof(v));

    /* Multi-head attention */
    float attn_out[BITNET_N_EMBD];
    memset(attn_out, 0, sizeof(attn_out));
    float attn_scale = 1.0f / sqrtf((float)BITNET_HEAD_DIM);

    for (int h = 0; h < BITNET_N_HEAD; h++) {
        int hs = h * BITNET_HEAD_DIM;

        /* Compute attention scores for all positions */
        float scores[BITNET_BLOCK_SIZE];
        for (int t = 0; t < seq_len; t++) {
            float dot = 0.0f;
            for (int j = 0; j < BITNET_HEAD_DIM; j++) {
                dot += q[hs + j] * ctx->kv_keys[t][hs + j];
            }
            scores[t] = dot * attn_scale;
        }

        /* Softmax */
        float probs[BITNET_BLOCK_SIZE];
        softmax(scores, probs, seq_len);

        /* Weighted sum of values */
        for (int j = 0; j < BITNET_HEAD_DIM; j++) {
            float s = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                s += probs[t] * ctx->kv_vals[t][hs + j];
            }
            attn_out[hs + j] = s;
        }
    }

    /* Output projection (ternary MVM) */
    float proj_out[BITNET_N_EMBD];
    ternary_mvm(block0_wo_weights, attn_out, proj_out,
                     BITNET_N_EMBD, BITNET_N_EMBD);

    /* Residual connection */
    for (int i = 0; i < BITNET_N_EMBD; i++) {
        x[i] = x_in[i] + proj_out[i];
    }

    /* -- MLP -- */
    float x_mid[BITNET_N_EMBD];
    memcpy(x_mid, x, sizeof(x_mid));

    /* RMSNorm before MLP */
    rmsnorm(x, xn, BITNET_N_EMBD);

    /* fc1 (N_EMBD -> MLP_DIM) */
    float h1[BITNET_MLP_DIM];
    ternary_mvm(block0_fc1_weights, xn, h1,
                     BITNET_N_EMBD, BITNET_MLP_DIM);

    /* ReLU^2 activation (matches microgpt-c) */
    float h2[BITNET_MLP_DIM];
    for (int i = 0; i < BITNET_MLP_DIM; i++) {
        h2[i] = h1[i] > 0 ? h1[i] * h1[i] : 0;
    }

    /* fc2 (MLP_DIM -> N_EMBD) */
    float mlp_out[BITNET_N_EMBD];
    ternary_mvm(block0_fc2_weights, h2, mlp_out,
                     BITNET_MLP_DIM, BITNET_N_EMBD);

    /* Residual connection */
    for (int i = 0; i < BITNET_N_EMBD; i++) {
        x[i] = x_mid[i] + mlp_out[i];
    }

    /* ---- Final RMSNorm ---- */
    rmsnorm(x, x, BITNET_N_EMBD);

    /* ---- LM Head (ternary MVM) ---- */
    ternary_mvm(bitnet_lm_head_weights,
                     x, logits_out, BITNET_N_EMBD, BITNET_VOCAB_SIZE);

    ctx->pos = seq_len;
}

/* ====================================================================
 * Token Generation
 * ==================================================================== */

int bitnet_generate(bitnet_ctx_t *ctx, int token_id, float temperature) {
    float logits[BITNET_VOCAB_SIZE];
    bitnet_forward(ctx, token_id, logits);

    if (temperature <= 0.0f) {
        /* Greedy: argmax */
        int best = 0;
        for (int i = 1; i < BITNET_VOCAB_SIZE; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    /* Temperature scaling + softmax */
    float inv_t = 1.0f / temperature;
    for (int i = 0; i < BITNET_VOCAB_SIZE; i++) logits[i] *= inv_t;

    float probs[BITNET_VOCAB_SIZE];
    softmax(logits, probs, BITNET_VOCAB_SIZE);

    /* Weighted random sampling */
    float r = rng_uniform(&ctx->rng_state);
    float cum = 0.0f;
    for (int i = 0; i < BITNET_VOCAB_SIZE; i++) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return BITNET_VOCAB_SIZE - 1;
}

/* ====================================================================
 * Token <-> Character Mapping
 * ==================================================================== */

char bitnet_token_to_char(int token_id) {
    if (token_id >= 0 && token_id < BITNET_VOCAB_SIZE) {
        return bitnet_id_to_char[token_id];
    }
    return '\0';
}

int bitnet_char_to_token(char c) {
    for (int i = 0; i < BITNET_VOCAB_SIZE - 1; i++) {
        if (bitnet_id_to_char[i] == c) return i;
    }
    return -1;
}

/* ====================================================================
 * Main: Demo / Benchmark
 * ==================================================================== */

#ifdef BITNET_GPT_MAIN
int main(void) {
    bitnet_ctx_t ctx;
    bitnet_init(&ctx);

    printf("BitNet-GPT Inference Engine\n");
    printf("  Architecture: %dL-%dH-%dE-%dM\n",
           BITNET_N_LAYER, BITNET_N_HEAD, BITNET_N_EMBD, BITNET_MLP_DIM);
    printf("  Vocab size: %d\n", BITNET_VOCAB_SIZE);
    printf("  Model size: %d bytes (%.1f KB)\n",
           BITNET_TOTAL_BYTES, BITNET_TOTAL_KB);
    printf("\n");

    /* Count ternary operations */
    int ternary_ops = 0;
    for (int i = 0; i < sizeof(bitnet_layers)/sizeof(bitnet_layers[0]); i++) {
        ternary_ops += bitnet_layers[i].n_inputs * bitnet_layers[i].n_outputs;
    }
    printf("  Ternary MVM ops per token: %d\n", ternary_ops);
    printf("  (No multiplications in inner loop!)\n\n");

    /* Generate names */
    float temperature = 0.5f;
    printf("--- Generated Names (temperature=%.1f) ---\n", temperature);

    for (int si = 0; si < 10; si++) {
        bitnet_reset(&ctx);
        int token_id = BITNET_BOS_TOKEN;
        char name[64];
        int len = 0;

        for (int pos = 0; pos < BITNET_BLOCK_SIZE; pos++) {
            token_id = bitnet_generate(&ctx, token_id, temperature);
            if (token_id == BITNET_BOS_TOKEN) break;
            char c = bitnet_token_to_char(token_id);
            if (c && len < 63) name[len++] = c;
        }
        name[len] = '\0';
        printf("  sample %2d: %s\n", si + 1, name);
    }

    /* Benchmark */
    printf("\n--- Benchmark ---\n");
    long long N = 1000000LL;
    long long emitted = 0;
    int token_id = BITNET_BOS_TOKEN;

    /* Simple timer using clock() */
    clock_t t0 = clock();

    for (long long i = 0; i < N; i++) {
        token_id = bitnet_generate(&ctx, token_id, 0.5f);
        if (token_id == BITNET_BOS_TOKEN) {
            bitnet_reset(&ctx);
            token_id = BITNET_BOS_TOKEN;
        }
        emitted++;
    }

    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("  bitnet-gpt ternary  %14.0f tok/sec\n", emitted / elapsed);
    printf("  (on this platform, with float32 activations)\n");

    return 0;
}
#endif
