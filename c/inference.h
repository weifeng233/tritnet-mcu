/*
 * BitNet-GPT Inference Engine
 * Ternary weight char-level GPT for MCU deployment
 *
 * Architecture: 1-layer transformer, 4 heads, 16-dim, 64-dim MLP
 * Weights: Ternary {-1, 0, +1} with base-3 packing (10 trits per uint16)
 * Activations: float32 (compact model, float is fine for MCU with FPU)
 *
 * Based on:
 *   - microgpt-c (pure C GPT, AVX2/FMA)
 *   - BitNetMCU (ternary quantization for MCU)
 *
 * Compile (x86 test): gcc -O3 -DBITNET_GPT_MAIN -o bitnet_gpt.exe bitnet_gpt_inference.c -lm
 * Compile (MCU):       Use your toolchain's float support
 */

#ifndef BITNET_GPT_INFERENCE_H
#define BITNET_GPT_INFERENCE_H

#include <stdint.h>

/* Forward declarations - model data comes from bitnet_gpt_model.h */

/* Inference context (statically allocated, zero heap) */
typedef struct {
    /* KV cache (float32 for precision) */
    float kv_keys[BITNET_BLOCK_SIZE][BITNET_N_EMBD];
    float kv_vals[BITNET_BLOCK_SIZE][BITNET_N_EMBD];

    /* Current position in sequence */
    int pos;

    /* RNG state for sampling */
    uint32_t rng_state;
} bitnet_ctx_t;

/* Initialize context (call once) */
void bitnet_init(bitnet_ctx_t *ctx);

/* Reset KV cache (call for each new sequence) */
void bitnet_reset(bitnet_ctx_t *ctx);

/* Generate next token given current token */
int bitnet_generate(bitnet_ctx_t *ctx, int token_id, float temperature);

/* Run full inference and return logits */
void bitnet_forward(bitnet_ctx_t *ctx, int token_id, float *logits_out);

/* Get character from token ID */
char bitnet_token_to_char(int token_id);

/* Get token ID from character */
int bitnet_char_to_token(char c);

#endif /* BITNET_GPT_INFERENCE_H */
