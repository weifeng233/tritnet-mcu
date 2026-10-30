"""
TritNet-MCU QAT Training Script (legacy PyTorch path)
===========================
PyTorch QAT (Quantization-Aware Training) for a ternary-weight char-level GPT.
Architecture mirrors microgpt-c: 1-layer transformer, 4 heads, 16-dim embeddings,
64-dim MLP, ReLU^2 activation, RMSNorm.

Weight quantization: Ternary {-1, 0, +1} using STE (Straight-Through Estimator)
Activation quantization: Per-token 8-bit

Based on:
  - microgpt-c (pure C GPT)
  - BitNetMCU (ternary quantization for MCU)

Usage:
  python examples/gpt_names/python/train_qat.py [--steps N] [--data PATH]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
import os
from datetime import datetime

# ============================================================================
# Configuration (matches microgpt-c exactly)
# ============================================================================

N_EMBD = 16
N_HEAD = 4
N_LAYER = 1
BLOCK_SIZE = 16
HEAD_DIM = N_EMBD // N_HEAD  # 4
MLP_DIM = 4 * N_EMBD          # 64

# ============================================================================
# BitNet Ternary Quantization
# ============================================================================

class TernaryQuantize(torch.autograd.Function):
    """Ternary weight quantization with Straight-Through Estimator (STE).

    Forward: quantize to {-1, 0, +1} using threshold = 0.5 * mean(|w|)
    Backward: pass gradient through unchanged (STE)
    """
    @staticmethod
    def forward(ctx, w):
        # Per-tensor threshold: 0.5 * mean(|w|)
        threshold = 0.5 * w.abs().mean()
        # Quantize: -1, 0, or +1
        w_q = torch.where(w > threshold, torch.ones_like(w),
                          torch.where(w < -threshold, -torch.ones_like(w),
                                      torch.zeros_like(w)))
        ctx.save_for_backward(w, threshold)
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        w, threshold = ctx.saved_tensors
        # STE: pass gradient through where |w| < 2*threshold
        mask = (w.abs() < 2 * threshold).float()
        return grad_output * mask


class BitLinear(nn.Module):
    """Linear layer with ternary weight quantization (QAT).

    During training: weights are quantized to {-1, 0, +1} via STE
    Activations are quantized to 8-bit per-token

    Scale factors are computed and returned for C inference calibration.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def activation_quant(self, x):
        """Per-token quantization to 8 bits (symmetric)."""
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        x_q = (x * scale).round().clamp_(-128, 127)
        return x_q, scale

    def forward(self, x):
        # Quantize activations to 8-bit
        x_norm = x  # RMSNorm is applied before this layer
        x_q, x_scale = self.activation_quant(x_norm)
        x_ste = x_norm + (x_q / x_scale - x_norm).detach()

        # Quantize weights to ternary
        w_q = TernaryQuantize.apply(self.weight)
        w_ste = self.weight + (w_q - self.weight).detach()

        # Linear
        y = F.linear(x_ste, w_ste)
        return y

    def get_quantized_weight(self):
        """Return the ternary quantized weight (for export)."""
        with torch.no_grad():
            threshold = 0.5 * self.weight.abs().mean()
            w_q = torch.where(self.weight > threshold, torch.ones_like(self.weight),
                              torch.where(self.weight < -threshold,
                                          -torch.ones_like(self.weight),
                                          torch.zeros_like(self.weight)))
        return w_q.cpu().numpy().astype(np.int8)

    def get_weight_scale(self):
        """Return the weight scale factor (for C inference)."""
        with torch.no_grad():
            return (1.0 / self.weight.abs().mean()).item()


# ============================================================================
# GPT Model
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-5)
        return x / rms


class BitNetAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = BitLinear(N_EMBD, N_EMBD)
        self.wk = BitLinear(N_EMBD, N_EMBD)
        self.wv = BitLinear(N_EMBD, N_EMBD)
        self.wo = BitLinear(N_EMBD, N_EMBD)

    def forward(self, x):
        B, T, C = x.shape

        # Q, K, V projections (ternary quantized)
        q = self.wq(x).view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        k = self.wk(x).view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        v = self.wv(x).view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / np.sqrt(HEAD_DIM)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, N_HEAD, T, T)

        # Causal mask: prevent attending to future positions
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)

        # Attention output
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, N_EMBD)
        return self.wo(out)


class BitNetMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = BitLinear(N_EMBD, MLP_DIM)
        self.fc2 = BitLinear(MLP_DIM, N_EMBD)

    def forward(self, x):
        h = self.fc1(x)
        # ReLU^2 activation (matches microgpt-c)
        h = F.relu(h) ** 2
        return self.fc2(h)


class BitNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = RMSNorm(N_EMBD)
        self.attn = BitNetAttention()
        self.ln2 = RMSNorm(N_EMBD)
        self.mlp = BitNetMLP()

    def forward(self, x):
        # Pre-norm attention with residual
        x = x + self.attn(self.ln1(x))
        # Pre-norm MLP with residual
        x = x + self.mlp(self.ln2(x))
        return x


class BitNetGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.ModuleList([BitNetBlock() for _ in range(N_LAYER)])
        self.ln_f = RMSNorm(N_EMBD)
        self.lm_head = BitLinear(N_EMBD, vocab_size)

        # Init embeddings with small values
        nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wpe.weight, mean=0.0, std=0.02)

    def forward(self, tokens):
        B, T = tokens.shape

        # Token + position embeddings
        tok_emb = self.wte(tokens)
        pos_ids = torch.arange(T, device=tokens.device)
        pos_emb = self.wpe(pos_ids)
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def get_all_quantized_weights(self):
        """Export all ternary weights and scales for C code generation."""
        result = {}

        # Embeddings (not quantized - stored as float)
        result['wte'] = self.wte.weight.detach().cpu().numpy()
        result['wpe'] = self.wpe.weight.detach().cpu().numpy()

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            prefix = f'block{i}'
            result[f'{prefix}_wq'] = block.attn.wq.get_quantized_weight()
            result[f'{prefix}_wq_scale'] = block.attn.wq.get_weight_scale()
            result[f'{prefix}_wk'] = block.attn.wk.get_quantized_weight()
            result[f'{prefix}_wk_scale'] = block.attn.wk.get_weight_scale()
            result[f'{prefix}_wv'] = block.attn.wv.get_quantized_weight()
            result[f'{prefix}_wv_scale'] = block.attn.wv.get_weight_scale()
            result[f'{prefix}_wo'] = block.attn.wo.get_quantized_weight()
            result[f'{prefix}_wo_scale'] = block.attn.wo.get_weight_scale()
            result[f'{prefix}_fc1'] = block.mlp.fc1.get_quantized_weight()
            result[f'{prefix}_fc1_scale'] = block.mlp.fc1.get_weight_scale()
            result[f'{prefix}_fc2'] = block.mlp.fc2.get_quantized_weight()
            result[f'{prefix}_fc2_scale'] = block.mlp.fc2.get_weight_scale()

        # LM head
        result['lm_head'] = self.lm_head.get_quantized_weight()
        result['lm_head_scale'] = self.lm_head.get_weight_scale()

        return result


# ============================================================================
# Dataset / Tokenizer
# ============================================================================

def load_names(filename):
    """Load names dataset (one name per line)."""
    with open(filename, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def build_tokenizer(names):
    """Build character-level tokenizer from names."""
    chars = sorted(set(''.join(names)))
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for c, i in char_to_id.items()}
    vocab_size = len(chars) + 1  # +1 for BOS token
    BOS = len(chars)
    return char_to_id, id_to_char, vocab_size, BOS


# ============================================================================
# Training
# ============================================================================

def train(args):
    print("=" * 60)
    print("BitNet-GPT Training (Ternary QAT)")
    print("=" * 60)

    # Load dataset
    names = load_names(args.data)
    np.random.shuffle(names)
    print(f"Loaded {len(names)} names")

    # Build tokenizer
    char_to_id, id_to_char, vocab_size, BOS = build_tokenizer(names)
    print(f"Vocab size: {vocab_size} ({len(char_to_id)} chars + BOS)")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BitNetGPT(vocab_size).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    ternary_params = 0
    float_params = 0
    for name, p in model.named_parameters():
        if 'weight' in name and ('wq' in name or 'wk' in name or 'wv' in name
                                 or 'wo' in name or 'fc1' in name or 'fc2' in name
                                 or 'lm_head' in name):
            if 'ln' not in name:  # RMSNorm has no params
                ternary_params += p.numel()
            else:
                float_params += p.numel()
        else:
            float_params += p.numel()
    print(f"Total parameters: {total_params}")
    print(f"  Ternary (quantized): {ternary_params}")
    print(f"  Float (embeddings): {float_params}")
    print(f"  Ternary storage: {ternary_params * 2 / 8 / 1024:.2f} KB (base-3 packing)")
    print(f"  Float storage: {float_params * 4 / 1024:.2f} KB")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), eps=1e-8)

    # Cosine LR schedule
    def lr_lambda(step):
        return 0.5 * (1.0 + np.cos(np.pi * step / args.steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    print(f"\nTraining for {args.steps} steps...")
    print("-" * 60)

    running_loss = 3.3
    best_loss = float('inf')

    for step in range(args.steps):
        model.train()

        # Pick a random name
        name = names[step % len(names)]

        # Build token sequence: [BOS, c1, c2, ..., cn, BOS]
        tokens = [BOS] + [char_to_id[c] for c in name] + [BOS]
        n = min(BLOCK_SIZE, len(tokens) - 1)

        # Truncate to BLOCK_SIZE
        tokens = tokens[:n + 1]

        # Forward pass (process full sequence in parallel with causal mask)
        tokens_tensor = torch.tensor([tokens[:-1]], device=device)  # (1, n)
        targets_tensor = torch.tensor(tokens[1:n+1], device=device)  # (n,)

        logits = model(tokens_tensor)  # (1, n, vocab_size)
        logits = logits.view(-1, vocab_size)  # (n, vocab_size)
        loss = F.cross_entropy(logits, targets_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Logging
        loss_val = loss.item()
        running_loss = running_loss * 0.99 + loss_val * 0.01

        if loss_val < best_loss:
            best_loss = loss_val

        if (step + 1) % 100 == 0 or step == 0 or step == args.steps - 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"step {step+1:5d}/{args.steps} | loss {loss_val:.4f} "
                  f"(avg {running_loss:.4f}) | lr {lr_now:.6f}")

    print("-" * 60)
    print(f"Training complete. Best loss: {best_loss:.4f}")

    # Save model checkpoint
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'char_to_id': char_to_id,
        'id_to_char': id_to_char,
        'BOS': BOS,
        'config': {
            'N_EMBD': N_EMBD,
            'N_HEAD': N_HEAD,
            'N_LAYER': N_LAYER,
            'BLOCK_SIZE': BLOCK_SIZE,
            'MLP_DIM': MLP_DIM,
        }
    }
    torch.save(checkpoint, args.output)
    print(f"Saved checkpoint to {args.output}")

    # Inference demo
    print("\n--- Inference ---")
    model.eval()
    temperature = args.temperature

    with torch.no_grad():
        for si in range(10):
            generated = [BOS]
            name = ''

            for pos in range(BLOCK_SIZE):
                tokens_tensor = torch.tensor([generated], device=device)
                logits = model(tokens_tensor)
                logits = logits[0, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                token_id = torch.multinomial(probs, 1).item()
                if token_id == BOS:
                    break
                generated.append(token_id)
                if token_id in id_to_char:
                    name += id_to_char[token_id]

            print(f"  sample {si+1:2d}: {name}")

    # Save quantized weights for export
    weights = model.get_all_quantized_weights()
    weights_file = args.output.replace('.pth', '_weights.npz')
    np.savez(weights_file, **{k: v for k, v in weights.items()})
    print(f"\nSaved quantized weights to {weights_file}")

    return model, checkpoint


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BitNet-GPT Ternary QAT Training')
    parser.add_argument('--data', type=str, default='examples/gpt_names/data/names.txt',
                        help='Training data file (one name per line)')
    parser.add_argument('--output', type=str, default='examples/gpt_names/modeldata/bitnet_gpt.pth',
                        help='Output checkpoint path')
    parser.add_argument('--steps', type=int, default=20000,
                        help='Number of training steps')
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Sampling temperature for inference demo')
    args = parser.parse_args()

    train(args)
