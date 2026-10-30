"""Layer-group sensitivity of absmean ternarization on Timestral-5M.

For each group of Linear weights (patch embedding, attention, gated projections, SwiGLU
MLP, quantile head) ternarize that group alone and measure the pinball loss on the same
fixed strokes, so the quantization plan can keep the delicate layers in fp32.

    python examples/timestral_sketch/python/timestral_sensitivity.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

EX = Path(__file__).resolve().parents[1]   # examples/timestral_sketch
ROOT = EX.parents[1]                          # repository root
sys.path.insert(0, str(EX / "reference"))

from timestral import TimestralForecaster  # noqa: E402

MODEL_DIR = EX / "modeldata" / "timestral-5m"
CTX, H, N = 256, 96, 24


def strokes() -> np.ndarray:
    rng = np.random.default_rng(7)
    out = np.empty((N, CTX + H), dtype=np.float32)
    t = np.arange(CTX + H)
    for i in range(N):
        kind = i % 4
        if kind == 0:
            y = 0.02 * t + rng.normal(0, 0.05, t.size) + np.convolve(
                rng.normal(0, 0.08, t.size), np.ones(9) / 9, mode="same")
        elif kind == 1:
            period = rng.uniform(40, 110)
            y = np.sin(2 * np.pi * t / period) * rng.uniform(0.5, 2.0) + rng.normal(0, 0.03, t.size)
        elif kind == 2:
            y = np.cumsum(rng.normal(0, 0.06, t.size))
        else:
            y = np.repeat(rng.uniform(-1, 2, (t.size + 24) // 24), 24)[: t.size]
        out[i] = y
    return out


def groups() -> dict[str, list[str]]:
    names = {
        "patch": ["representation.", "head."],
        "attn": [".attns."],
        "gate": ["._gate."],
        "mlp": [".mlp."],
    }
    out = {k: [] for k in names}
    out["all"] = list(names)
    return out, names


def group_of(name: str) -> str:
    if name.startswith(("representation.", "head.")):
        return "patch"
    if "._gate." in name:
        return "gate"
    if ".attns." in name:
        return "attn"
    if ".mlp." in name:
        return "mlp"
    return "other"


def ternarize_(model, keep: set[str]) -> np.ndarray:
    """Per-row absmean; ternarize groups not in `keep`. Returns trits (n, )."""
    hist = [0, 0, 0]
    with torch.no_grad():
        for name, mod in model.named_modules():
            g = group_of(name)
            if not isinstance(mod, torch.nn.Linear) or g in keep:
                continue
            w = mod.weight.data
            rows = w.detach().numpy().astype(np.float32).reshape(w.shape[0], -1)
            scales = np.mean(np.abs(rows), axis=1)
            trits = np.where(rows > scales[:, None] / 2, 1.0, np.where(rows < -scales[:, None] / 2, -1.0, 0.0))
            mod.weight.data.copy_(torch.from_numpy((trits * scales[:, None]).reshape(w.shape)))
            hist[0] += int((trits == -1).sum())
            hist[1] += int((trits == 0).sum())
            hist[2] += int((trits == 1).sum())
    return np.asarray(hist)


def run(fc, data) -> dict:
    ctx, tgt = data[:, :CTX], data[:, CTX:]
    f = fc.forecast(ctx, prediction_length=H, clip_nonnegative=False)
    q = f.quantiles[:, 0].numpy()
    lv = np.asarray(f.quantile_levels)
    e = tgt[..., None] - q
    med = q[..., len(lv) // 2]
    return {
        "pinball": float(np.mean(np.maximum(lv * e, (lv - 1) * e))),
        "mase": float(np.mean(np.abs(med - tgt)) / np.mean(np.abs(np.diff(ctx, axis=-1)))),
        "mae": float(np.mean(np.abs(med - tgt))),
    }


def main() -> None:
    data = strokes()
    fc = TimestralForecaster.from_pretrained(MODEL_DIR)
    print(f"fp32           {run(fc, data)}")
    allg, names = groups()
    for label, keep in [
        ("all ternary", set()),
        ("keep patch", {"patch"}),
        ("keep patch+mlp", {"patch", "mlp"}),
        ("keep patch+mlp+gate", {"patch", "mlp", "gate"}),
        ("keep patch+attn", {"patch", "attn"}),
        ("keep head only", {"patch"}),
    ]:
        m = copy.deepcopy(fc.model)
        hist = ternarize_(m, keep)
        out = run(TimestralForecaster(m), data)
        tot = hist.sum()
        print(f"{label:14s} {out}  -1 {hist[0]/tot:.1%} 0 {hist[1]/tot:.1%} +1 {hist[2]/tot:.1%}")


if __name__ == "__main__":
    main()
