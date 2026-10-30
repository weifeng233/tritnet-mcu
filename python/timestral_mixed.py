"""Which parts of Timestral-5M must stay fp32 to survive 1.6 bits?

Ternarizes the fine-tuned checkpoint group by group and scores the same 24 app strokes
through the reference model, so the answer is measured rather than assumed. The packed
artifact can carry a mixture (fp32 matrices + pinned trits), so a group that has to stay
dense is a deployment option, not a dead end.

    python python/timestral_mixed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "python"))

from timestral import TimestralForecaster  # noqa: E402
from timestral_metrics import summarize  # noqa: E402

QAT_DIR = ROOT / "modeldata" / "timestral_qat"
CTX, H = 256, 96
LEVELS = np.asarray([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99], dtype=np.float32)

GROUPS = {
    "head": ("head.",),
    "patch": ("representation.",),
    "attn": (".attns.",),
    "mlp": (".mlp.",),
}


def group_of(name: str) -> str:
    for group, prefixes in GROUPS.items():
        if any(p in name for p in prefixes):
            return group
    return "norms"


def ternarize_(model: torch.nn.Module, keep: set[str]) -> int:
    """Per-row absmean on every Linear outside `keep`; returns the ternary parameter count."""
    n = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.Linear) or group_of(name) in keep:
                continue
            w = mod.weight.data
            rows = w.reshape(w.shape[0], -1)
            scale = rows.abs().mean(dim=1, keepdim=True)
            t = 0.5 * scale
            trit = torch.where(rows > t, 1.0, torch.where(rows < -t, -1.0, 0.0))
            mod.weight.data.copy_((trit * scale).reshape(w.shape))
            n += w.numel()
    return n


def main() -> None:
    corpus = np.fromfile(ROOT / "modeldata" / "timestral_strokes.bin", dtype=np.float32).reshape(-1, CTX + H)
    ctx_np, truth = corpus[:, :CTX], corpus[:, CTX:]
    ctx = torch.from_numpy(ctx_np)
    base = TimestralForecaster.from_pretrained(QAT_DIR)
    torch.set_num_threads(4)

    rows = []
    for label, keep in [
        ("fp32 (QAT latents)", {"head", "patch", "attn", "mlp", "norms"}),
        ("all ternary", set()),
        ("keep head", {"head"}),
        ("keep head+patch", {"head", "patch"}),
        ("keep head+patch+attn", {"head", "patch", "attn"}),
    ]:
        fc = TimestralForecaster.from_pretrained(QAT_DIR)
        tern = ternarize_(fc.model, keep)
        with torch.inference_mode():
            q = fc.forecast(ctx, prediction_length=H, clip_nonnegative=False).quantiles[:, 0].numpy()
        s = summarize(q, truth, ctx_np, LEVELS)
        bits = (tern * 1.6 + (5208960 - tern) * 32) / 5208960
        rows.append((label, tern, bits, s))
        print(f"{label:22s} ternary {tern/1e6:5.2f} M  ~{bits:5.2f} bits/param  "
              f"pinball {s['pinball']:.4f}  mae {s['mae']:.4f}  cov90 {s['cov90']:.3f}")


if __name__ == "__main__":
    main()
