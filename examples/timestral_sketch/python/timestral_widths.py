"""What does a quantile-width scale do to the deployed artifact?

The app's forecast is `median + gamma * (q - median)`; this script sweeps gamma on the
24-stroke corpus and reports pinball and coverage, for the fp32 engine and the ternary one.
Used to justify (or reject) `python`-side width calibration.

    python examples/timestral_sketch/python/timestral_widths.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from timestral_metrics import summarize  # noqa: E402

EX = Path(__file__).resolve().parents[1]   # examples/timestral_sketch
ROOT = EX.parents[1]                          # repository root
LEVELS = np.asarray([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99], dtype=np.float32)


def main() -> None:
    corpus = np.fromfile(EX / "modeldata" / "timestral_strokes.bin", dtype=np.float32).reshape(-1, 352)
    ctx, truth = corpus[:, :256], corpus[:, 256:384]
    for name in ("fp32", "ternary"):
        q = np.fromfile(EX / "modeldata" / f"timestral_forecast_{name}.bin", dtype=np.float32).reshape(-1, 96, 21)
        med = q[..., 10:11]
        print(name)
        for g in (1.0, 1.2, 1.4, 1.6, 2.0, 2.5):
            s = summarize(med + g * (q - med), truth, ctx, LEVELS)
            print(f"  gamma {g:4.1f}  pinball {s['pinball']:.4f}  mae {s['mae']:.4f}  cov90 {s['cov90']:.3f}")


if __name__ == "__main__":
    main()
