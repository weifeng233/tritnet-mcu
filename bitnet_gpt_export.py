"""
BitNet-GPT Export Script
========================
Exports ternary-quantized GPT model to a C header file for MCU inference.

Weight packing: Base-3 encoding (10 trits per uint16 word)
  - Maps {-1, 0, +1} to {1, 2, 0} for efficient unpacking
  - Packs 10 trits into 16 bits using base-3
  - C unpacking: multiply by 3, check bit 17 for zero, bit 16 for sign

Based on BitNetMCU's exportquant.py packing scheme.

Usage:
  python bitnet_gpt_export.py --checkpoint modeldata/bitnet_gpt.pth
"""

import torch
import numpy as np
import argparse
import os
import sys
from datetime import datetime

# ============================================================================
# Import training config
# ============================================================================
sys.path.insert(0, os.path.dirname(__file__))
from bitnet_gpt_train import N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, HEAD_DIM, MLP_DIM, BitNetGPT


# ============================================================================
# Ternary Weight Packing (Base-3)
# ============================================================================

def pack_row(row):
    """Pack a single row of ternary weights into uint16 words.

    Each row is padded to a multiple of 10, then packed using base-3.
    This matches how the C inference unpacks row-by-row.

    Returns: numpy array of uint16
    """
    flat = row.flatten().astype(np.int32)
    n = len(flat)

    # Pad to multiple of 10
    if n % 10 != 0:
        pad_size = 10 - (n % 10)
        flat = np.pad(flat, (0, pad_size), constant_values=0)
        n = len(flat)

    # Map {-1, 0, +1} to {1, 2, 0}
    trit_values = np.where(flat == 1, 0, np.where(flat == -1, 1, 2)).astype(np.uint32)

    # Pack 10 trits into each uint16
    n_packed = n // 10
    packed = np.zeros(n_packed, dtype=np.uint16)

    for i in range(n_packed):
        start = i * 10
        chunk = trit_values[start:start + 10]

        # Base-3 packing: MSB first
        value = 0
        for t in range(10):
            value = value * 3 + chunk[t]

        # Scale to use full uint16 range: (value * 65536 + 59048) // 59049
        packed[i] = (value * 65536 + 59048) // 59049

    return packed


def pack_ternary_matrix(weights, n_inputs_per_row):
    """Pack a 2D weight matrix row by row.

    Each row of n_inputs_per_row elements is packed independently.
    This matches how the C inference engine unpacks: packed_per_row = ceil(n_in / 10).

    Returns: flat numpy array of uint16 (all rows concatenated)
    """
    n_rows = weights.shape[0]
    packed_per_row = (n_inputs_per_row + 9) // 10
    all_packed = np.zeros(n_rows * packed_per_row, dtype=np.uint16)

    for r in range(n_rows):
        row_packed = pack_row(weights[r, :])
        all_packed[r * packed_per_row:(r + 1) * packed_per_row] = row_packed

    return all_packed


def verify_packing_matrix(packed, original, n_inputs_per_row):
    """Verify that unpacking reproduces the original weights (row by row).

    Uses the SAME unpacking algorithm as the C inference engine:
    multiply-by-3 and bit extraction (BitNetMCU method).
    """
    n_rows = original.shape[0]
    packed_per_row = (n_inputs_per_row + 9) // 10
    total_mismatches = 0

    for r in range(n_rows):
        row_orig = original[r, :].flatten().astype(np.int32)
        n_out = len(row_orig)
        row_packed = packed[r * packed_per_row:(r + 1) * packed_per_row]
        unpacked = np.zeros(n_out, dtype=np.int32)

        for i in range(len(row_packed)):
            chunk_val = int(row_packed[i])

            # BitNetMCU's unpacking: multiply by 3, extract from bits 16-17
            for t in range(10):
                idx = i * 10 + t
                if idx >= n_out:
                    break

                chunk_val *= 3
                if not (chunk_val & 0x20000):  # bit 17 = 0 means non-zero
                    if chunk_val & 0x10000:    # bit 16 = sign
                        unpacked[idx] = -1
                    else:
                        unpacked[idx] = 1
                else:
                    unpacked[idx] = 0
                chunk_val &= 0xFFFF  # Clear upper bits

        total_mismatches += np.sum(unpacked[:n_out] != row_orig)

    return total_mismatches


# ============================================================================
# C Header Generation
# ============================================================================

def generate_c_header(checkpoint_path, output_path):
    """Generate C header file from checkpoint."""

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Reconstruct model
    vocab_size = checkpoint['vocab_size']
    char_to_id = checkpoint['char_to_id']
    id_to_char = checkpoint['id_to_char']
    BOS = checkpoint['BOS']

    model = BitNetGPT(vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get quantized weights
    weights = model.get_all_quantized_weights()

    print(f"Vocab size: {vocab_size}")
    print(f"Generating C header: {output_path}")

    with open(output_path, 'w') as f:
        # Header
        f.write("/*\n")
        f.write(f" * BitNet-GPT Model Weights\n")
        f.write(f" * Generated: {datetime.now()}\n")
        f.write(f" * Architecture: {N_LAYER}L-{N_HEAD}H-{N_EMBD}E-{MLP_DIM}M\n")
        f.write(f" * Vocab size: {vocab_size}\n")
        f.write(f" * Ternary weights: {{-1, 0, +1}} (base-3 packed)\n")
        f.write(f" *\n")
        f.write(f" * From: {checkpoint_path}\n")
        f.write(f" */\n\n")

        f.write("#ifndef BITNET_GPT_MODEL_H\n")
        f.write("#define BITNET_GPT_MODEL_H\n\n")
        f.write("#include <stdint.h>\n\n")

        # Model configuration
        f.write("/* Model configuration */\n")
        f.write(f"#define BITNET_VOCAB_SIZE  {vocab_size}\n")
        f.write(f"#define BITNET_N_EMBD     {N_EMBD}\n")
        f.write(f"#define BITNET_N_HEAD     {N_HEAD}\n")
        f.write(f"#define BITNET_N_LAYER    {N_LAYER}\n")
        f.write(f"#define BITNET_BLOCK_SIZE {BLOCK_SIZE}\n")
        f.write(f"#define BITNET_HEAD_DIM   {HEAD_DIM}\n")
        f.write(f"#define BITNET_MLP_DIM    {MLP_DIM}\n\n")

        # BOS token
        f.write(f"#define BITNET_BOS_TOKEN  {BOS}\n\n")

        # Character mapping table
        f.write("/* Character to ID mapping */\n")
        f.write("static const char bitnet_id_to_char[BITNET_VOCAB_SIZE] = {\n")
        for i in range(vocab_size):
            if i in id_to_char:
                c = id_to_char[i]
                # Escape special chars
                if c == '\\': c = '\\\\'
                elif c == '\'': c = '\\\''
                elif c == '\"': c = '\\"'
                elif c == '\n': c = '\\n'
                elif c == '\t': c = '\\t'
                f.write(f"    '{c}',  /* {i} */\n")
            else:
                f.write(f"    '\\0', /* {i} (BOS) */\n")
        f.write("};\n\n")

        # Embeddings (float32 - not quantized)
        wte = weights['wte']
        f.write("/* Token embeddings (float32, not quantized) */\n")
        f.write(f"static const float bitnet_wte[{vocab_size}][{N_EMBD}] = {{\n")
        for i in range(vocab_size):
            vals = ', '.join(f'{v:.6e}f' for v in wte[i])
            f.write(f"    {{ {vals} }},\n")
        f.write("};\n\n")

        wpe = weights['wpe']
        f.write("/* Position embeddings (float32, not quantized) */\n")
        f.write(f"static const float bitnet_wpe[{BLOCK_SIZE}][{N_EMBD}] = {{\n")
        for i in range(BLOCK_SIZE):
            vals = ', '.join(f'{v:.6e}f' for v in wpe[i])
            f.write(f"    {{ {vals} }},\n")
        f.write("};\n\n")

        # Ternary weights for each layer
        total_bits = 0
        for layer in range(N_LAYER):
            prefix = f'block{layer}'
            f.write(f"/* === Transformer Block {layer} === */\n\n")

            for wname in ['wq', 'wk', 'wv', 'wo', 'fc1', 'fc2']:
                key = f'{prefix}_{wname}'
                w = weights[key]
                scale = weights[f'{key}_scale']

                # Determine shape and n_inputs for row-by-row packing
                if wname in ['wq', 'wk', 'wv', 'wo']:
                    shape_str = f"{N_EMBD}x{N_EMBD}"
                    n_in, n_out = N_EMBD, N_EMBD
                elif wname == 'fc1':
                    shape_str = f"{MLP_DIM}x{N_EMBD}"
                    n_in, n_out = N_EMBD, MLP_DIM
                elif wname == 'fc2':
                    shape_str = f"{N_EMBD}x{MLP_DIM}"
                    n_in, n_out = MLP_DIM, N_EMBD

                packed = pack_ternary_matrix(w, n_in)

                # Verify
                mismatches = verify_packing_matrix(packed, w, n_in)
                if mismatches > 0:
                    print(f"WARNING: {key} has {mismatches} packing mismatches!")

                f.write(f"/* {wname}: {shape_str}, scale={scale:.6f} */\n")
                f.write(f"static const float {key}_scale = {scale:.8e}f;\n")
                f.write(f"static const uint16_t {key}_weights[{len(packed)}] = {{\n")
                for i in range(0, len(packed), 8):
                    chunk = packed[i:i+8]
                    vals = ', '.join(f'0x{v:04x}' for v in chunk)
                    f.write(f"    {vals},\n")
                f.write("};\n\n")

                total_bits += len(packed) * 16
                print(f"  {key}: {w.shape} -> {len(packed)} uint16 words "
                      f"({len(packed)*2} bytes)")

        # LM head
        f.write("/* === LM Head === */\n\n")
        lm_w = weights['lm_head']
        lm_scale = weights['lm_head_scale']
        packed = pack_ternary_matrix(lm_w, N_EMBD)
        mismatches = verify_packing_matrix(packed, lm_w, N_EMBD)
        if mismatches > 0:
            print(f"WARNING: lm_head has {mismatches} packing mismatches!")

        f.write(f"/* lm_head: {vocab_size}x{N_EMBD}, scale={lm_scale:.6f} */\n")
        f.write(f"static const float bitnet_lm_head_scale = {lm_scale:.8e}f;\n")
        f.write(f"static const uint16_t bitnet_lm_head_weights[{len(packed)}] = {{\n")
        for i in range(0, len(packed), 8):
            chunk = packed[i:i+8]
            vals = ', '.join(f'0x{v:04x}' for v in chunk)
            f.write(f"    {vals},\n")
        f.write("};\n\n")
        total_bits += len(packed) * 16

        print(f"  lm_head: {lm_w.shape} -> {len(packed)} uint16 words")

        # Summary
        f.write("/* Model size summary */\n")
        f.write(f"#define BITNET_TOTAL_BITS    {total_bits}\n")
        f.write(f"#define BITNET_TOTAL_BYTES   {total_bits // 8}\n")
        f.write(f"#define BITNET_TOTAL_KB      ({total_bits} / 8 / 1024)\n\n")

        # Weight lookup table (for C inference)
        f.write("/* Weight pointer table for layer iteration */\n")
        f.write("typedef struct {\n")
        f.write("    const uint16_t *weights;\n")
        f.write("    float scale;\n")
        f.write("    int n_inputs;\n")
        f.write("    int n_outputs;\n")
        f.write("    int n_packed_words;\n")
        f.write("} bitnet_layer_t;\n\n")

        f.write(f"static const bitnet_layer_t bitnet_layers[] = {{\n")
        for layer in range(N_LAYER):
            prefix = f'block{layer}'
            for wname in ['wq', 'wk', 'wv', 'wo', 'fc1', 'fc2']:
                key = f'{prefix}_{wname}'
                w = weights[key]
                if wname in ['wq', 'wk', 'wv', 'wo']:
                    n_in, n_out = N_EMBD, N_EMBD
                elif wname == 'fc1':
                    n_in, n_out = N_EMBD, MLP_DIM
                elif wname == 'fc2':
                    n_in, n_out = MLP_DIM, N_EMBD

                packed_w = pack_ternary_matrix(w, n_in)
                f.write(f"    {{ {key}_weights, {key}_scale, {n_in}, {n_out}, {len(packed_w)} }},\n")

        # lm_head
        n_packed_lm = len(packed)
        f.write(f"    {{ bitnet_lm_head_weights, bitnet_lm_head_scale, {N_EMBD}, {vocab_size}, {n_packed_lm} }},\n")
        f.write("};\n\n")

        f.write("#endif /* BITNET_GPT_MODEL_H */\n")

    print(f"\n=== Export Summary ===")
    print(f"Total ternary weights: {total_bits} bits ({total_bits/8/1024:.2f} KB)")
    print(f"Float embeddings: {(vocab_size * N_EMBD + BLOCK_SIZE * N_EMBD) * 4 / 1024:.2f} KB")
    total_flash = total_bits / 8 + (vocab_size * N_EMBD + BLOCK_SIZE * N_EMBD) * 4
    print(f"Total Flash: {total_flash / 1024:.2f} KB")
    print(f"Output: {output_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export BitNet-GPT to C header')
    parser.add_argument('--checkpoint', type=str, default='modeldata/bitnet_gpt.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='bitnet_gpt_model.h',
                        help='Output C header file')
    args = parser.parse_args()

    generate_c_header(args.checkpoint, args.output)
