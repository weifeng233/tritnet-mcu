#!/usr/bin/env python3
"""
train.py — PyTorch training script for MCU-MLP models

Trains MLP models for:
  1. Classification (multi-class, binary, multi-label)
  2. Motor control (regression: direct control signal or PID gain scheduling)
  3. Anomaly detection (binary classification)

Exports trained weights to model.h via export_weights.py

Usage:
  # Classification example (vibration sensor fault detection)
  python train.py --task classify --data vibration_data.csv \
      --input-dim 8 --hidden-dims 32,32 --output-dim 4 \
      --epochs 500 --lr 3e-4 --act relu2,tanh

  # Motor control (direct MLP output)
  python train.py --task motor --data motor_data.csv \
      --input-dim 12 --hidden-dims 64,64,16 --output-dim 1 \
      --epochs 1000 --lr 1e-3 --act relu2,relu2,tanh

  # Anomaly detection
  python train.py --task anomaly --data normal_data.csv \
      --input-dim 8 --hidden-dims 32,32 --output-dim 1 \
      --epochs 500 --lr 3e-4 --act relu2,sigmoid
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# ====================================================================
# Activation mapping: CLI string -> PyTorch module
# ====================================================================

ACT_MAP = {
    'relu':    nn.ReLU,
    'relu2':   lambda: Lambda(lambda x: torch.where(x > 0, x * x, torch.zeros_like(x))),
    'tanh':    nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'none':    nn.Identity,
}


class Lambda(nn.Module):
    """Wraps an arbitrary function as an nn.Module."""
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


# ====================================================================
# MLP Model
# ====================================================================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activations):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            act_name = activations[i] if i < len(activations) else 'relu2'
            layers.append(ACT_MAP[act_name]())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        # Output activation (last in activations list, or identity)
        if len(activations) > len(hidden_dims):
            layers.append(ACT_MAP[activations[-1]]())

        self.net = nn.Sequential(*layers)

        # Initialize weights (microgpt-c style: N(0, 0.02))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ====================================================================
# Data loading utilities
# ====================================================================

def load_csv_data(path, input_dim, output_dim, delimiter=','):
    """Load CSV data: last column(s) are targets, rest are features."""
    data = np.loadtxt(path, delimiter=delimiter, dtype=np.float32)
    X = data[:, :input_dim]
    Y = data[:, input_dim:input_dim + output_dim]
    return X, Y


def normalize_features(X_train, X_val=None):
    """Z-score normalization: returns mean, std arrays."""
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_n = ((X_val - mean) / std) if X_val is not None else None
    return X_train_n, X_val_n, mean, std


# ====================================================================
# Training loops per task type
# ====================================================================

def train_classification(model, X_train, Y_train, X_val, Y_val,
                          epochs, lr, batch_size, device):
    """Multi-class classification: cross-entropy loss."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_train_t = torch.tensor(X_train, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.long, device=device)

    n_samples = X_train.shape[0]
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            x = X_train_t[idx]
            y = Y_train_t[idx]

            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_loss = total_loss / n_batches
            msg = f"epoch {epoch+1:4d}/{epochs} | loss {avg_loss:.4f}"

            if X_val is not None:
                val_acc = evaluate_classification(model, X_val, Y_val, device)
                msg += f" | val_acc {val_acc:.2%}"
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            print(msg)

    return model


def train_regression(model, X_train, Y_train, X_val, Y_val,
                      epochs, lr, batch_size, device):
    """Regression (motor control): MSE loss + optional control effort penalty."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_train_t = torch.tensor(X_train, device=device)
    Y_train_t = torch.tensor(Y_train, device=device)

    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            x = X_train_t[idx]
            y = Y_train_t[idx]

            pred = model(x)
            loss = nn.MSELoss()(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_loss = total_loss / n_batches
            msg = f"epoch {epoch+1:4d}/{epochs} | mse {avg_loss:.6f}"

            if X_val is not None:
                val_mse = evaluate_regression(model, X_val, Y_val, device)
                msg += f" | val_mse {val_mse:.6f}"
            print(msg)

    return model


def train_anomaly(model, X_train, epochs, lr, batch_size, device):
    """Anomaly detection: train on normal data, use autoencoder-style loss.
    Simplified: train to output 0.0 for normal data."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_train_t = torch.tensor(X_train, device=device)
    Y_train_t = torch.zeros(X_train.shape[0], 1, device=device)  # target: 0 = normal

    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            x = X_train_t[idx]

            logit = model(x)
            pred = torch.sigmoid(logit)
            loss = nn.BCEWithLogitsLoss()(logit, Y_train_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_loss = total_loss / n_batches
            print(f"epoch {epoch+1:4d}/{epochs} | bce {avg_loss:.6f}")

    return model


# ====================================================================
# Evaluation utilities
# ====================================================================

def evaluate_classification(model, X, Y, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, device=device)
        logits = model(X_t)
        preds = logits.argmax(dim=1).cpu().numpy()
    return (preds == Y).mean()


def evaluate_regression(model, X, Y, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, device=device)
        pred = model(X_t).cpu().numpy()
    return np.mean((pred - Y) ** 2)


# ====================================================================
# Main
# ====================================================================

def parse_dims(s):
    return [int(x) for x in s.split(',')]


def parse_acts(s):
    return [x.strip() for x in s.split(',')]


def main():
    parser = argparse.ArgumentParser(description='Train MLP for MCU deployment')
    parser.add_argument('--task', required=True, choices=['classify', 'motor', 'anomaly'],
                        help='Task type: classify | motor | anomaly')
    parser.add_argument('--data', required=True, help='Training data CSV file')
    parser.add_argument('--val-data', default=None, help='Validation data CSV file')
    parser.add_argument('--input-dim', type=int, required=True, help='Number of input features')
    parser.add_argument('--hidden-dims', required=True, help='Hidden layer dims, comma-separated (e.g., 64,64,16)')
    parser.add_argument('--output-dim', type=int, default=1, help='Number of output classes/values')
    parser.add_argument('--act', default='relu2', help='Activations per layer, comma-separated (e.g., relu2,relu2,tanh)')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('--output', default='model.h', help='Output header file path')
    parser.add_argument('--quantize', action='store_true', help='Also generate int8 quantized weights')
    parser.add_argument('--device', default='auto', help='Device: auto | cpu | cuda')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Parse architecture
    hidden_dims = parse_dims(args.hidden_dims)
    activations = parse_acts(args.act)
    all_dims = [args.input_dim] + hidden_dims + [args.output_dim]

    print(f"Architecture: {' -> '.join(str(d) for d in all_dims)}")
    print(f"Activations: {activations}")

    # Load data
    X, Y = load_csv_data(args.data, args.input_dim, args.output_dim)
    print(f"Loaded {X.shape[0]} samples, {args.input_dim} features -> {args.output_dim} outputs")

    # Normalize
    X_norm, _, feat_mean, feat_std = normalize_features(X)

    # Split (if no separate val file)
    if args.val_data:
        X_val, Y_val = load_csv_data(args.val_data, args.input_dim, args.output_dim)
        X_val_n = (X_val - feat_mean) / feat_std
    else:
        n = X_norm.shape[0]
        split = int(n * 0.85)
        X_val_n = X_norm[split:]
        Y_val = Y[split:]
        X_norm = X_norm[:split]
        Y = Y[:split]

    # Create model
    model = MLP(args.input_dim, hidden_dims, args.output_dim, activations)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Train
    print("\n--- Training ---")
    if args.task == 'classify':
        model = train_classification(model, X_norm, Y, X_val_n, Y_val,
                                      args.epochs, args.lr, args.batch_size, device)
    elif args.task == 'motor':
        model = train_regression(model, X_norm, Y, X_val_n, Y_val,
                                  args.epochs, args.lr, args.batch_size, device)
    elif args.task == 'anomaly':
        model = train_anomaly(model, X_norm, args.epochs, args.lr, args.batch_size, device)

    # Export weights
    print(f"\n--- Exporting to {args.output} ---")
    sys.path.insert(0, str(Path(__file__).parent))
    from export_weights import export_model_header

    export_model_header(
        model=model,
        hidden_dims=hidden_dims,
        activations=activations,
        output_path=args.output,
        feat_mean=feat_mean,
        feat_std=feat_std,
        task=args.task,
        quantize=args.quantize,
    )

    print(f"Done. Wrote {args.output} ({n_params} parameters)")
    if args.quantize:
        print(f"Quantized version: {args.output.replace('.h', '_q.h')}")


if __name__ == '__main__':
    main()
