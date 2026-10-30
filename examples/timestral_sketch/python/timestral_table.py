"""Per-family table for docs/TIMESTRAL.md: fp32 vs ternary on the same strokes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from timestral_metrics import KIND_NAMES, by_kind  # noqa: E402

EX = Path(__file__).resolve().parents[1]   # examples/timestral_sketch
ROOT = EX.parents[1]                          # repository root
LEVELS = np.asarray([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99], dtype=np.float32)


def main() -> None:
    corpus = np.fromfile(EX / "modeldata" / "timestral_strokes.bin", dtype=np.float32).reshape(-1, 352)
    ctx, truth = corpus[:, :256], corpus[:, 256:384]
    names = [k for k in ("fp32", "ternary")
             if (EX / "modeldata" / f"timestral_forecast_{k}.bin").exists()]
    qs = {k: np.fromfile(EX / "modeldata" / f"timestral_forecast_{k}.bin", dtype=np.float32).reshape(-1, 96, 21)
          for k in names}
    tables = {k: by_kind(qs[k], truth, ctx, LEVELS) for k in names}
    print("| family | strokes | fp32 pinball | ternary pinball | ratio |")
    print("| --- | --- | --- | --- | --- |")
    for i, kind in enumerate(KIND_NAMES):
        a, b = tables["fp32"][i], tables["ternary"][i]
        print(f"| {kind} | {a['strokes']} | {a['pinball']:.4f} | {b['pinball']:.4f} | {b['pinball'] / a['pinball']:.2f}x |")
    print()
    for k in names:
        s = {x: np.mean([r[x] for r in tables[k]]) for x in ("pinball", "mae", "mase", "cov90")}
        print(k, {x: round(float(v), 4) for x, v in s.items()})


if __name__ == "__main__":
    main()
