"""End-to-end parity: the same strokes through PyTorch and through the Zig engine.

`timestral-cli eval --dump` writes the stroke corpus it scored; this script loads that
corpus, forecasts it with the reference PyTorch forecaster (clampsym on, exactly what the
app requests), and compares against the Zig engine's batch output. It also recomputes the
metrics in NumPy, so the numbers printed by the two engines can be checked against a third
implementation.

    python python/timestral_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))

from timestral import TimestralForecaster  # noqa: E402
from timestral_metrics import summarize  # noqa: E402

MODEL_DIR = ROOT / "modeldata" / "timestral-5m"
CORPUS = ROOT / "modeldata" / "timestral_strokes.bin"
ZIG_FP32 = ROOT / "modeldata" / "timestral_forecast_fp32.bin"
CTX, H = 256, 96



def main() -> None:
    corpus = np.fromfile(CORPUS, dtype=np.float32).reshape(-1, CTX + H)
    ctx, truth = corpus[:, :CTX], corpus[:, CTX:]
    fc = TimestralForecaster.from_pretrained(MODEL_DIR)
    torch.set_num_threads(4)
    with torch.inference_mode():
        q_torch = fc.forecast(torch.from_numpy(ctx), prediction_length=H, clip_nonnegative=False).quantiles[:, 0].numpy()
    levels = np.asarray(fc.quantile_levels, dtype=np.float32)

    q_zig = np.fromfile(ZIG_FP32, dtype=np.float32).reshape(len(corpus), H, len(levels))
    diff = np.abs(q_torch - q_zig)
    scale = np.max(np.abs(q_torch))
    print(f"strokes {len(corpus)}  horizon {H}  quantiles {len(levels)}")
    print(f"max |torch - zig|      {diff.max():.6f}   (forecast magnitude {scale:.3f})")
    print(f"mean |torch - zig|     {diff.mean():.6f}")
    print(f"relative max error     {diff.max() / scale:.2e}")

    a = summarize(q_torch, truth, ctx, levels)
    b = summarize(q_zig, truth, ctx, levels)
    print(f"torch metrics  {a}")
    print(f"zig   metrics  {b}")
    rel = {k: abs(a[k] - b[k]) / max(abs(a[k]), 1e-9) for k in a}
    print("relative metric diff " + "  ".join(f"{k} {v:.1e}" for k, v in rel.items()))
    ok = diff.max() / scale < 1e-3 and max(rel.values()) < 1e-2
    print("parity:", "OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
