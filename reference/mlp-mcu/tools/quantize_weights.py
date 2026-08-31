#!/usr/bin/env python3
"""
quantize_weights.py — INT8 per-channel symmetric weight quantization

Takes a model.h file (float32 weights) and generates a quantized version
with int8 weights and per-channel float32 scale factors.

Quantization scheme:
  For each output channel i:
    max_abs = max(|W_float[i][j]|) for all j
    scale[i] = max_abs / 127.0
    W_quant[i][j] = clamp(round(W_float[i][j] / scale[i]), -128, 127)

Reconstruction:
  W_approx[i][j] = W_quant[i][j] * scale[i]

Typical accuracy loss: < 1% for well-trained models.

Usage:
  python quantize_weights.py --input model.h --output model_q.h [--check]

  --check: verify quantization error is below 5% and print statistics
"""

import argparse
import re
import numpy as np


# ====================================================================
# Parse C arrays from model.h
# ====================================================================

def parse_float_array_2d(content, name):
    """Parse 'static const float name[ROWS][COLS] = { ... };' from C header."""
    pattern = rf'static const float {name}\[(\d+)\]\[(\d+)\]\s*=\s*\{{([^;]+)\}};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None
    rows, cols = int(match.group(1)), int(match.group(2))
    data_str = match.group(3)

    # Parse nested braces
    row_pattern = r'\{([^}]+)\}'
    row_matches = re.findall(row_pattern, data_str)

    arr = np.zeros((rows, cols), dtype=np.float32)
    for r, row_str in enumerate(row_matches):
        vals = [float(v.strip().rstrip('f')) for v in row_str.split(',')]
        arr[r, :len(vals)] = vals

    return arr


def parse_float_array_1d(content, name):
    """Parse 'static const float name[N] = { ... };' from C header."""
    pattern = rf'static const float {name}\[(\d+)\]\s*=\s*\{{([^;]+)\}};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None
    n = int(match.group(1))
    data_str = match.group(2)
    vals = [float(v.strip().rstrip('f')) for v in data_str.split(',')]
    arr = np.array(vals, dtype=np.float32)
    return arr


# ====================================================================
# Quantization
# ====================================================================

def quantize_channel(W_row):
    """Quantize a single output channel to int8 with symmetric scale."""
    max_abs = np.abs(W_row).max()
    scale = max_abs / 127.0 if max_abs > 1e-10 else 1e-10
    W_q = np.round(W_row / scale).clip(-128, 127).astype(np.int8)
    return W_q, scale


def quantize_matrix(W):
    """Quantize entire weight matrix: per-channel symmetric int8."""
    n_out, n_in = W.shape
    W_q = np.zeros((n_out, n_in), dtype=np.int8)
    scales = np.zeros(n_out, dtype=np.float32)
    for i in range(n_out):
        W_q[i], scales[i] = quantize_channel(W[i])
    return W_q, scales


def quantization_error(W, W_q, scales):
    """Compute max and mean absolute quantization error."""
    n_out, n_in = W.shape
    errors = np.zeros_like(W)
    for i in range(n_out):
        W_approx = W_q[i].astype(np.float32) * scales[i]
        errors[i] = np.abs(W[i] - W_approx)
    return errors.max(), errors.mean()


# ====================================================================
# C header rewriting
# ====================================================================

def format_float_2d(arr, name):
    rows, cols = arr.shape
    lines = [f'static const float {name}[{rows}][{cols}] = {{']
    for r in range(rows):
        vals = ', '.join(f'{v:.8e}f' for v in arr[r])
        comma = ',' if r < rows - 1 else ''
        lines.append(f'  {{ {vals} }}{comma}')
    lines.append('};')
    return '\n'.join(lines)


def format_float_1d(arr, name):
    n = arr.shape[0]
    vals = ', '.join(f'{v:.8e}f' for v in arr)
    return f'static const float {name}[{n}] = {{ {vals} }};'


def format_int8_2d(arr, name):
    rows, cols = arr.shape
    lines = [f'static const int8_t {name}[{rows}][{cols}] = {{']
    for r in range(rows):
        vals = ', '.join(f'{int(v)}' for v in arr[r])
        comma = ',' if r < rows - 1 else ''
        lines.append(f'  {{ {vals} }}{comma}')
    lines.append('};')
    return '\n'.join(lines)


def quantize_model_header(input_path, output_path, check=False):
    """
    Read model.h, quantize all weight matrices, write model_q.h.

    Adds to the output:
      - MLP_USE_INT8_WEIGHTS 1
      - int8_t mlp_wN_q[][] arrays
      - float mlp_wN_scales[] arrays
    """
    with open(input_path, 'r') as f:
        content = f.read()

    # Find all weight arrays: mlp_w0, mlp_w1, ...
    weight_pattern = r'static const float (mlp_w(\d+))\[(\d+)\]\[(\d+)\]'
    matches = re.findall(weight_pattern, content)

    if not matches:
        print(f"Error: no weight arrays (mlp_wN) found in {input_path}")
        return

    # Parse and quantize each weight matrix
    quant_info = []  # (layer_idx, W_q, scales, bias_array_name)
    stats = []

    for name, layer_idx_str, rows, cols in matches:
        layer_idx = int(layer_idx_str)
        W = parse_float_array_2d(content, name)
        if W is None:
            print(f"Warning: could not parse {name}, skipping")
            continue

        W_q, scales = quantize_matrix(W)
        max_err, mean_err = quantization_error(W, W_q, scales)

        quant_info.append((layer_idx, W_q, scales))
        stats.append({
            'layer': layer_idx,
            'shape': W.shape,
            'params': W.size,
            'max_err': max_err,
            'mean_err': mean_err,
            'max_err_pct': max_err / (np.abs(W).max() + 1e-10) * 100,
        })

    if check:
        print("\n=== Quantization Quality Report ===\n")
        total_params = 0
        for s in stats:
            total_params += s['params']
            print(f"Layer {s['layer']}: shape {s['shape']}, "
                  f"max_err={s['max_err']:.6f} ({s['max_err_pct']:.2f}%), "
                  f"mean_err={s['mean_err']:.6f}")
        print(f"\nTotal parameters: {total_params}")
        overall_max = max(s['max_err_pct'] for s in stats)
        print(f"Overall max error: {overall_max:.2f}%")
        if overall_max > 5.0:
            print("WARNING: Max error > 5%. Model accuracy may be significantly degraded.")
        elif overall_max > 1.0:
            print("NOTE: Max error > 1%. Minor accuracy loss expected.")
        else:
            print("Quantization quality: GOOD (< 1% error)")
        print()

    # Generate output: copy original content, add quantized arrays
    output_lines = []

    # Add include guard and header
    output_lines.append('/*')
    output_lines.append(f' * {output_path} — INT8 quantized model weights')
    output_lines.append(f' * Generated from: {input_path}')
    output_lines.append(f' */')
    output_lines.append('')
    output_lines.append(f'#ifndef MODEL_Q_H')
    output_lines.append(f'#define MODEL_Q_H')
    output_lines.append('')
    output_lines.append('#include <stdint.h>')
    output_lines.append('#include "mlp_platform.h"')
    output_lines.append('')

    # Copy original content (skip header comments and include guard)
    # Find the first #define or #include after the initial comment block
    start_match = re.search(r'(#define|#include)', content)
    if start_match:
        output_lines.append(content[start_match.start():])
    else:
        output_lines.append(content)

    output_lines.append('')

    # Add quantized arrays
    output_lines.append('/* ===== INT8 quantized weights (75%% Flash savings) ===== */')
    output_lines.append('#undef  MLP_USE_INT8_WEIGHTS')
    output_lines.append('#define MLP_USE_INT8_WEIGHTS 1')
    output_lines.append('')

    for layer_idx, W_q, scales in quant_info:
        output_lines.append(format_int8_2d(W_q, f'mlp_w{layer_idx}_q'))
        output_lines.append('')
        output_lines.append(format_float_1d(scales, f'mlp_w{layer_idx}_scales'))
        output_lines.append('')

    # Scale pointer array
    n_layers = len(quant_info)
    s_ptrs = ', '.join(f'(const float*)mlp_w{i}_scales' for i in range(n_layers))
    output_lines.append(f'const float *const mlp_scales[{n_layers}] = {{ {s_ptrs} }};')
    output_lines.append('')

    output_lines.append(f'#endif /* MODEL_Q_H */')
    output_lines.append('')

    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"Wrote quantized model to {output_path}")


# ====================================================================
# CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='Quantize model.h to int8')
    parser.add_argument('--input', required=True, help='Input model.h (float32)')
    parser.add_argument('--output', default=None, help='Output model_q.h (int8)')
    parser.add_argument('--check', action='store_true', help='Print quantization quality report')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.replace('.h', '_q.h')

    quantize_model_header(args.input, args.output, check=args.check)


if __name__ == '__main__':
    main()
