#!/usr/bin/env python3
"""
export_weights.py — Export trained PyTorch MLP weights to model.h

Generates a C header file containing:
  - Model architecture config (#define constants)
  - Weight arrays (const float, stored in Flash/ROM)
  - Bias arrays (const float)
  - Feature normalization parameters
  - Optional: int8 quantized weights (via quantize_weights.py)

Usage:
  from export_weights import export_model_header
  export_model_header(model, hidden_dims, activations, 'model.h')

  Or standalone:
  python export_weights.py --checkpoint model.pt --output model.h --quantize
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ====================================================================
# Activation name mapping
# ====================================================================

ACT_TO_C = {
    'relu':    'MLP_ACT_RELU',
    'relu2':   'MLP_ACT_RELU2',
    'tanh':    'MLP_ACT_TANH',
    'sigmoid': 'MLP_ACT_SIGMOID',
    'none':    'MLP_ACT_NONE',
}


def act_name_to_c(name):
    return ACT_TO_C.get(name.strip().lower(), 'MLP_ACT_NONE')


# ====================================================================
# Extract weights from PyTorch model
# ====================================================================

def extract_weights(model):
    """Extract linear layer weights and biases as numpy arrays."""
    layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            W = m.weight.data.cpu().numpy()  # [out_features, in_features]
            b = m.bias.data.cpu().numpy()    # [out_features]
            layers.append((W, b))
    return layers


# ====================================================================
# Quantization (optional)
# ====================================================================

def quantize_weight_matrix(W):
    """
    Per-channel symmetric int8 quantization.
    Returns: W_q (int8 array), scales (float array, one per output channel)
    """
    n_out, n_in = W.shape
    W_q = np.zeros_like(W, dtype=np.int8)
    scales = np.zeros(n_out, dtype=np.float32)

    for i in range(n_out):
        max_abs = np.abs(W[i]).max()
        scale = max_abs / 127.0 if max_abs > 1e-10 else 1e-10
        scales[i] = scale
        w_q = np.round(W[i] / scale).clip(-128, 127).astype(np.int8)
        W_q[i] = w_q

    return W_q, scales


# ====================================================================
# C header generation
# ====================================================================

def format_float_array_2d(arr, name, dtype='float'):
    """Format a 2D numpy array as a C static const array."""
    rows, cols = arr.shape
    lines = [f'static const {dtype} {name}[{rows}][{cols}] = {{']
    for r in range(rows):
        vals = ', '.join(f'{v:.8e}f' for v in arr[r])
        comma = ',' if r < rows - 1 else ''
        lines.append(f'  {{ {vals} }}{comma}')
    lines.append('};')
    return '\n'.join(lines)


def format_float_array_1d(arr, name, dtype='float'):
    """Format a 1D numpy array as a C static const array."""
    n = arr.shape[0]
    vals = ', '.join(f'{v:.8e}f' for v in arr)
    return f'static const {dtype} {name}[{n}] = {{ {vals} }};'


def format_int8_array_2d(arr, name):
    """Format a 2D int8 numpy array as a C static const array."""
    rows, cols = arr.shape
    lines = [f'static const int8_t {name}[{rows}][{cols}] = {{']
    for r in range(rows):
        vals = ', '.join(f'{int(v)}' for v in arr[r])
        comma = ',' if r < rows - 1 else ''
        lines.append(f'  {{ {vals} }}{comma}')
    lines.append('};')
    return '\n'.join(lines)


def export_model_header(model, hidden_dims, activations, output_path,
                         feat_mean=None, feat_std=None,
                         task='classify', quantize=False):
    """
    Export a trained PyTorch MLP to a C header file.

    Args:
        model: trained nn.Module
        hidden_dims: list of hidden layer dimensions
        activations: list of activation names per layer
        output_path: path to write the .h file
        feat_mean: feature normalization means (numpy array)
        feat_std: feature normalization stds (numpy array)
        task: task type string (for documentation)
        quantize: also generate int8 quantized weights
    """
    layers = extract_weights(model)
    n_layers = len(layers)

    input_dim = layers[0][0].shape[1]
    output_dim = layers[-1][0].shape[0]
    layer_dims = [W.shape[0] for W, _ in layers]

    # Build output file
    lines = []
    lines.append('/*')
    lines.append(f' * model.h — Auto-generated MCU-MLP model weights')
    lines.append(f' * Task: {task}')
    lines.append(f' * Architecture: {input_dim} -> {" -> ".join(str(d) for d in layer_dims)}')
    lines.append(f' * Parameters: {sum(W.size + b.size for W, b in layers)}')
    lines.append(f' *')
    lines.append(f' * DO NOT EDIT — regenerate with train.py')
    lines.append(f' */')
    lines.append('')
    lines.append('#ifndef MODEL_H')
    lines.append('#define MODEL_H')
    lines.append('')
    lines.append('#include <stdint.h>')
    lines.append('#include "mlp_platform.h"')
    lines.append('')

    # Architecture config
    lines.append('/* Model architecture */')
    lines.append(f'#define MLP_INPUT_DIM    {input_dim}')
    lines.append(f'#define MLP_OUTPUT_DIM   {output_dim}')
    lines.append(f'#define MLP_N_LAYERS     {n_layers}')
    all_dims_str = ', '.join(str(d) for d in layer_dims)
    lines.append(f'#define MLP_LAYER_DIMS   {{ {all_dims_str} }}')
    acts_str = ', '.join(act_name_to_c(a) for a in activations[:n_layers])
    lines.append(f'#define MLP_LAYER_ACTS   {{ {acts_str} }}')
    max_hidden = max(layer_dims[:-1]) if len(layer_dims) > 1 else layer_dims[0]
    lines.append(f'#define MLP_MAX_HIDDEN   {max_hidden}')
    lines.append('')

    # Float32 weights
    lines.append('/* ===== Float32 weights (stored in Flash/ROM) ===== */')
    lines.append('')
    for i, (W, b) in enumerate(layers):
        lines.append(format_float_array_2d(W, f'mlp_w{i}'))
        lines.append('')
        lines.append(format_float_array_1d(b, f'mlp_b{i}'))
        lines.append('')

    # Weight and bias pointer arrays (for mlp_forward_f32)
    lines.append('/* Weight/bias pointer arrays for generic forward pass */')
    w_ptrs = ', '.join(f'(const float*)mlp_w{i}' for i in range(n_layers))
    b_ptrs = ', '.join(f'(const float*)mlp_b{i}' for i in range(n_layers))
    lines.append(f'const float *const mlp_weights[{n_layers}] = {{ {w_ptrs} }};')
    lines.append(f'const float *const mlp_biases[{n_layers}]  = {{ {b_ptrs} }};')
    lines.append('')

    # Feature normalization
    if feat_mean is not None and feat_std is not None:
        lines.append('/* Feature normalization (z-score) */')
        lines.append(format_float_array_1d(feat_mean, 'clf_input_mean'))
        lines.append(format_float_array_1d(feat_std, 'clf_input_std'))
        lines.append('')

    # Int8 quantized weights (optional)
    if quantize:
        lines.append('/* ===== Int8 quantized weights ===== */')
        lines.append('#define MLP_USE_INT8_WEIGHTS 1')
        lines.append('')
        all_scales = []
        for i, (W, b) in enumerate(layers):
            W_q, scales = quantize_weight_matrix(W)
            lines.append(format_int8_array_2d(W_q, f'mlp_w{i}_q'))
            lines.append('')
            lines.append(format_float_array_1d(scales, f'mlp_w{i}_scales'))
            lines.append('')
            all_scales.append(scales)

        # Scale pointer array
        s_ptrs = ', '.join(f'(const float*)mlp_w{i}_scales' for i in range(n_layers))
        lines.append(f'const float *const mlp_scales[{n_layers}] = {{ {s_ptrs} }};')
        lines.append('')

    lines.append('#endif /* MODEL_H */')
    lines.append('')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


# ====================================================================
# Standalone CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='Export MLP weights to model.h')
    parser.add_argument('--checkpoint', required=True, help='PyTorch checkpoint (.pt)')
    parser.add_argument('--output', default='model.h', help='Output header file')
    parser.add_argument('--quantize', action='store_true', help='Also generate int8 weights')
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model = ckpt['model']
    hidden_dims = ckpt['hidden_dims']
    activations = ckpt['activations']
    feat_mean = ckpt.get('feat_mean')
    feat_std = ckpt.get('feat_std')
    task = ckpt.get('task', 'classify')

    export_model_header(model, hidden_dims, activations, args.output,
                         feat_mean, feat_std, task, args.quantize)

    print(f"Wrote {args.output}")


if __name__ == '__main__':
    main()
