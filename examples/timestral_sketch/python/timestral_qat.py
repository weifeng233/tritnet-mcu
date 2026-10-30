"""Quantization-aware training for Timestral-5M with this project's ternary quantizer.

Naive post-training ternarization costs the model most of its accuracy (see
`examples/timestral_sketch/python/timestral_sensitivity.py`): absmean row scales at 1.6 bits per weight are far from
what a pretrained fp32 transformer expects. This script fine-tunes the released checkpoint
with the framework's exact quantizer in the forward pass:

    scale = mean(|W_row|)          per output row, like framework/quant.zig
    trit  = +1 past scale/2, -1 below -scale/2, else 0
    W_q   = trit * scale           what the packed artifact deploys

Gradients reach the fp32 *latent* weights through a straight-through estimator, and (as in
BitNetMCU / the framework's `maskGrads`) they are masked to |W| < scale, so saturated
weights stop chasing the quantizer. The loss is the model's own pinball loss over the 21
quantiles on a corpus of drawn-looking strokes — the same distribution the app receives.

The fine-tuned checkpoint is saved as `model.safetensors` + `config.json` in the output
directory; `timestral-cli quantize` then produces the deployable artifact from it.

    python examples/timestral_sketch/python/timestral_qat.py --steps 1200 --batch 32
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

EX = Path(__file__).resolve().parents[1]   # examples/timestral_sketch
ROOT = EX.parents[1]                          # repository root
sys.path.insert(0, str(EX / "reference"))

from timestral import TimestralForecaster  # noqa: E402

MODEL_DIR = EX / "modeldata" / "timestral-5m"
OUT_DIR = EX / "modeldata" / "timestral_qat"
CTX = 256


# ---------------------------------------------------------------------------
# the framework's ternary quantizer, in torch
# ---------------------------------------------------------------------------

_QUANT = {"on": False, "masks": []}


def ternarize_rows(w: torch.Tensor) -> torch.Tensor:
    """Per-row absmean ternary weights: the exact rule of framework/quant.zig, vectorized."""
    scale = w.abs().mean(dim=-1, keepdim=True)
    t = 0.5 * scale
    trit = torch.where(w > t, 1.0, torch.where(w < -t, -1.0, 0.0))
    return trit * scale, t


def quantized_linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """STE forward: deploy ternary weights, backprop through the fp32 latents."""
    if not _QUANT["on"] or self.weight is None:
        return torch.nn.functional.linear(x, self.weight, self.bias)
    q, t = ternarize_rows(self.weight)
    _QUANT["masks"].append((self.weight, t.detach()))
    out = torch.nn.functional.linear(x, self.weight + (q - self.weight).detach(), self.bias)
    return out


def install_quantizer() -> None:
    nn.Linear.forward = quantized_linear_forward


def mask_gradients() -> None:
    """Zero gradients where |W| >= 2*t, the framework's STE mask."""
    for w, t in _QUANT["masks"]:
        if w.grad is not None:
            w.grad.mul_((w.abs() < 2 * t).to(w.grad.dtype))
    _QUANT["masks"].clear()


# ---------------------------------------------------------------------------
# stroke corpus
# ---------------------------------------------------------------------------


def _smooth_curve(rng: np.random.Generator, n: int) -> np.ndarray:
    """Cubic interpolation through a few random control points: what a hand draws."""
    k = rng.integers(3, 7)
    xs = np.linspace(0, 1, k)
    ys = rng.uniform(-1.2, 2.0, k)
    t = np.linspace(0, 1, n)
    idx = np.clip(np.searchsorted(xs, t) - 1, 0, k - 2)
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    m0 = np.zeros_like(y0)
    m1 = np.zeros_like(y1)
    # Catmull-Rom style tangents so the curve does not corner at the knots
    m0 = np.where(idx > 0, (ys[idx + 1] - ys[np.maximum(idx - 1, 0)]) / 2, 0.0)
    m1 = np.where(idx + 2 < k, (ys[np.minimum(idx + 2, k - 1)] - ys[idx]) / 2, 0.0)
    h = x1 - x0
    u = (t - x0) / h
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


def stroke_corpus(count: int, t: int, seed: int) -> np.ndarray:
    """`count` drawn-looking series of `t` steps, roughly on a [-2, 3] canvas."""
    rng = np.random.default_rng(seed)
    out = np.empty((count, t), dtype=np.float32)
    tt = np.arange(t, dtype=np.float64)
    kinds = 8
    for i in range(count):
        kind = i % kinds
        if kind == 0:  # smooth hand-drawn curve
            y = _smooth_curve(rng, t) + rng.normal(0, 0.02, t)
            y = y * rng.uniform(0.6, 1.8) + rng.uniform(-0.5, 1.0)
        elif kind == 1:  # sinusoid: a period the stroke keeps, a level that wanders
            period = rng.uniform(24, 140)
            phase = rng.uniform(0, 2 * math.pi)
            amp = rng.uniform(0.3, 2.2)
            level = rng.uniform(-0.5, 1.5)
            drift = rng.normal(0, 0.004, t).cumsum()
            y = level + drift + amp * np.sin(2 * np.pi * tt / period + phase) + rng.normal(0, 0.02, t)
        elif kind == 2:  # trend + smoothed noise
            noise = np.convolve(rng.normal(0, 1, t), np.ones(9) / 9, mode="same")
            y = rng.uniform(-0.02, 0.02) * tt + 0.35 * noise + rng.uniform(0, 1.5)
        elif kind == 3:  # random walk
            y = np.cumsum(rng.normal(0, 0.07, t)) + rng.uniform(-0.5, 0.5)
        elif kind == 4:  # steps / plateaus
            y = np.empty(t)
            pos = 0
            while pos < t:
                ln = int(rng.integers(10, 40))
                y[pos : pos + ln] = rng.uniform(-0.8, 2.2)
                pos += ln
            y = y + rng.normal(0, 0.01, t)
        elif kind == 5:  # mean-reverting AR(1)
            phi = rng.uniform(0.6, 0.99)
            mu = rng.uniform(-0.5, 2.0)
            x = mu
            y = np.empty(t)
            for j in range(t):
                x = mu + phi * (x - mu) + rng.normal(0, 0.09)
                y[j] = x
        elif kind == 6:  # AR(2) with a seasonal term
            y = np.empty(t)
            x1 = x2 = 0.0
            for j in range(t):
                x = 1.3 * x1 - 0.5 * x2 + rng.normal(0, 0.06)
                y[j] = x + 0.4 * math.sin(2 * math.pi * j / rng.uniform(20, 60))
                x2, x1 = x1, x
        else:  # heartbeat / spike train
            y = rng.normal(0, 0.01, t)
            for _ in range(int(rng.integers(3, 9))):
                c = int(rng.integers(10, t - 10))
                w = int(rng.integers(2, 6))
                y[c : c + w] += rng.uniform(0.5, 2.5)
        out[i] = np.clip(y, -20, 20)
    return out


def raw_forecast(fc: TimestralForecaster, x: torch.Tensor, h: int, symmetric: bool = True) -> torch.Tensor:
    """`TimestralForecaster.forecast` without its `inference_mode` decorator, so QAT can
    backprop through it. Same passes (context and its negation) and the same capping."""
    inputs, mask = fc._as_context(x, None)
    q = fc._forecast_raw(inputs, mask, h)
    if symmetric:
        mirrored = fc._forecast_raw(-inputs, mask, h)
        q = 0.5 * (q + (-mirrored).flip(-1))
    return q[:, 0]


def pinball(q: torch.Tensor, y: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss; q (B, H, Q), y (B, H)."""
    e = y[..., None] - q
    return torch.maximum(levels * e, (levels - 1) * e).mean()


def context_target(corpus: np.ndarray, idx: np.ndarray, ctx: int, h: int):
    rows = corpus[idx]
    return torch.from_numpy(rows[:, :ctx]), torch.from_numpy(rows[:, ctx : ctx + h])


def evaluate(fc: TimestralForecaster, corpus: np.ndarray, ctx: int, h: int, batch: int = 16) -> dict:
    fc.model.eval()
    levels = torch.tensor(fc.quantile_levels, dtype=torch.float32)
    tot_p, tot_mae, tot_cov, n = 0.0, 0.0, 0.0, 0
    with torch.inference_mode():
        for s in range(0, len(corpus), batch):
            idx = np.arange(s, min(s + batch, len(corpus)))
            x, y = context_target(corpus, idx, ctx, h)
            q = raw_forecast(fc, x, h)
            tot_p += pinball(q, y, levels).item() * len(idx)
            med = q[..., len(levels) // 2]
            tot_mae += (med - y).abs().mean().item() * len(idx)
            lo, hi = q[..., 1], q[..., -2]
            tot_cov += ((y >= lo) & (y <= hi)).float().mean().item() * len(idx)
            n += len(idx)
    return {"pinball": tot_p / n, "mae": tot_mae / n, "cov90": tot_cov / n}


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--h", type=int, default=96)
    ap.add_argument("--train", type=int, default=3072, help="training strokes")
    ap.add_argument("--evaln", type=int, default=128, help="held-out strokes")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--probe", action="store_true", help="time 5 steps and stop")
    ap.add_argument("--mask", action="store_true", help="apply the STE gradient mask (|W| < scale)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    install_quantizer()

    print(f"loading {MODEL_DIR}", flush=True)
    fc = TimestralForecaster.from_pretrained(MODEL_DIR)
    model = fc.model
    levels = torch.tensor(fc.quantile_levels, dtype=torch.float32)
    params = sum(p.numel() for p in model.parameters())
    print(f"params {params:,}")

    train = stroke_corpus(args.train, CTX + args.h, seed=11)
    heldout = stroke_corpus(args.evaln, CTX + args.h, seed=999)

    fp32 = evaluate(fc, heldout, CTX, args.h)
    print(f"fp32 held-out {fp32}", flush=True)

    _QUANT["on"] = True
    ptq = evaluate(fc, heldout, CTX, args.h)
    print(f"ptq  held-out {ptq}  (absmean per-row, no training)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    rng = np.random.default_rng(2026)

    model.train()
    t0 = time.perf_counter()
    running = 0.0
    for step in range(args.steps):
        idx = rng.integers(0, len(train), args.batch)
        x, y = context_target(train, idx, CTX, args.h)
        _QUANT["masks"].clear()
        q = raw_forecast(fc, x, args.h, symmetric=False)
        loss = pinball(q, y, levels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.mask:
            mask_gradients()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running += loss.item()
        if (step + 1) % 50 == 0 or step == 0:
            dt = (time.perf_counter() - t0) / (step + 1)
            print(f"step {step + 1:4d}/{args.steps}  loss {running / (50 if step else 1):.4f}  {dt * 1000:.0f} ms/step", flush=True)
            running = 0.0
        if args.probe and step >= 4:
            break

    if args.probe:
        return

    model.eval()
    qat = evaluate(fc, heldout, CTX, args.h)
    print(f"qat  held-out {qat}", flush=True)
    gap_ptq = ptq["pinball"] - fp32["pinball"]
    recovered = 0.0 if gap_ptq <= 0 else (ptq["pinball"] - qat["pinball"]) / gap_ptq
    print(f"pinball: fp32 {fp32['pinball']:.5f} | ptq {ptq['pinball']:.5f} | qat {qat['pinball']:.5f} "
          f"| qat recovers {100 * recovered:.1f}% of the ptq gap", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    _QUANT["on"] = False
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items() if not k.endswith("inv_freq")}
    from safetensors.torch import save_file

    save_file(state, str(args.out / "model.safetensors"))
    (args.out / "config.json").write_text((MODEL_DIR / "config.json").read_text())
    report = {
        "steps": args.steps, "batch": args.batch, "lr": args.lr, "h": args.h,
        "train_strokes": args.train, "heldout_strokes": args.evaln,
        "fp32": fp32, "ptq": ptq, "qat": qat,
        "qat_recovers_pct": 100 * recovered,
    }
    (args.out / "qat_report.json").write_text(json.dumps(report, indent=2) + "\n")
    size = (args.out / "model.safetensors").stat().st_size
    print(f"saved {args.out} ({size} B, {size * 8 / params:.2f} bits/weight across all tensors)")


if __name__ == "__main__":
    main()
