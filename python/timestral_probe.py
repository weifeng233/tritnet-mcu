"""Quick probe: how much does naive absmean ternarization cost Timestral-5M?

Runs the reference (PyTorch) forecaster on a fixed set of drawn-looking strokes and
reports pinball loss / MASE for several weight-quantization variants. Used to pick the
quantizer the Zig exporter implements.

    python python/timestral_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))

from timestral import TimestralForecaster  # noqa: E402

MODEL_DIR = ROOT / "modeldata" / "timestral-5m"
CTX = 256
H = 96
N = 24


def strokes() -> np.ndarray:
    """N drawn-looking strokes, (N, CTX + H): smooth, noisy, trending, blocky."""
    rng = np.random.default_rng(7)
    out = np.empty((N, CTX + H), dtype=np.float32)
    t = np.arange(CTX + H)
    for i in range(N):
        kind = i % 4
        if kind == 0:  # trend + noise
            y = 0.02 * t + rng.normal(0, 0.05, t.size) + np.convolve(
                rng.normal(0, 0.08, t.size), np.ones(9) / 9, mode="same")
        elif kind == 1:  # sinusoid
            period = rng.uniform(40, 110)
            y = np.sin(2 * np.pi * t / period) * rng.uniform(0.5, 2.0) + rng.normal(0, 0.03, t.size)
        elif kind == 2:  # random walk
            y = np.cumsum(rng.normal(0, 0.06, t.size))
        else:  # steps
            y = np.repeat(rng.uniform(-1, 2, (t.size + 24) // 24), 24)[: t.size]
        out[i] = y
    return out


def pinball(q: np.ndarray, y: np.ndarray, levels: np.ndarray) -> float:
    """Mean pinball loss; q (B, H, Q), y (B, H)."""
    e = y[..., None] - q
    return float(np.mean(np.maximum(levels * e, (levels - 1) * e)))


def mase(med: np.ndarray, y: np.ndarray, ctx: np.ndarray) -> float:
    naive = np.mean(np.abs(np.diff(ctx, axis=-1)))
    return float(np.mean(np.abs(med - y)) / max(naive, 1e-8))


def ternarize_(model, mode: str) -> dict:
    """Ternarize every Linear weight in place; returns the trit histogram."""
    hist = np.zeros(3, dtype=np.int64)
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.Linear) or mod.weight is None:
                continue
            w = mod.weight.data
            flat = w.detach().numpy().astype(np.float32)
            if mode == "per_tensor":
                scale = np.mean(np.abs(flat))
                trits = np.where(flat > scale / 2, 1.0, np.where(flat < -scale / 2, -1.0, 0.0))
                q = trits * scale
            elif mode in ("per_row", "per_row_ls"):
                rows = flat.reshape(-1, flat.shape[-1])
                scales = np.mean(np.abs(rows), axis=1)
                trits = np.where(rows > scales[:, None] / 2, 1.0,
                                 np.where(rows < -scales[:, None] / 2, -1.0, 0.0))
                if mode == "per_row_ls":
                    # least-squares optimal scale per row: <W,T>/<T,T>
                    denom = np.maximum((trits * trits).sum(1), 1.0)
                    scales = (rows * trits).sum(1) / denom
                q = (trits * scales[:, None]).reshape(flat.shape)
            else:
                raise ValueError(mode)
            hist[0] += int((q == 0).sum() + (flat == 0).sum()) * 0
            hist[0] += int(((trits if mode == "per_tensor" else trits) == -1).sum())
            hist[1] += int(((trits if mode == "per_tensor" else trits) == 0).sum())
            hist[2] += int(((trits if mode == "per_tensor" else trits) == 1).sum())
            mod.weight.data.copy_(torch.from_numpy(q).to(w.dtype))
    return {"neg": int(hist[0]), "zero": int(hist[1]), "pos": int(hist[2])}


def run(fc: TimestralForecaster, data: np.ndarray) -> dict:
    ctx, tgt = data[:, :CTX], data[:, CTX:]
    f = fc.forecast(ctx, prediction_length=H, clip_nonnegative=False)
    q = f.quantiles[:, 0].numpy()
    levels = np.asarray(f.quantile_levels)
    med = q[..., len(levels) // 2]
    return {
        "pinball": pinball(q, tgt, levels),
        "mase": mase(med, tgt, ctx),
        "mae": float(np.mean(np.abs(med - tgt))),
        "cov90": float(np.mean((tgt >= q[..., 2]) & (tgt <= q[..., -3]))),
    }


def main() -> None:
    data = strokes()
    fc = TimestralForecaster.from_pretrained(MODEL_DIR)
    print(f"params {sum(p.numel() for p in fc.model.parameters()):,}")
    ref = run(fc, data)
    print(f"fp32      {ref}")

    import copy
    for mode in ("per_tensor", "per_row", "per_row_ls"):
        m = copy.deepcopy(fc.model)
        fc2 = TimestralForecaster(m)
        hist = ternarize_(m, mode)
        out = run(fc2, data)
        tot = sum(hist.values())
        print(f"{mode:10s} {out}  trits -1 {hist['neg']/tot:.1%} 0 {hist['zero']/tot:.1%} +1 {hist['pos']/tot:.1%}")


if __name__ == "__main__":
    main()
