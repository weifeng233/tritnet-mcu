"""Forecast metrics shared by the parity check and the report.

One definition of pinball loss, MAE, MASE and interval coverage, computed in NumPy from
`(strokes, horizon, quantiles)` arrays, so the Python side can score the reference model
and the Zig engine's output with the same code.
"""

from __future__ import annotations

import numpy as np

KIND_NAMES = ("drawn curve", "sinusoid", "random walk", "trend+noise", "plateaus")


def pinball(q: np.ndarray, truth: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Per-stroke mean pinball loss; `q` (N, H, Q), `truth` (N, H)."""
    e = truth[..., None] - q
    return np.maximum(levels * e, (levels - 1) * e).mean(axis=(1, 2))


def mae(q: np.ndarray, truth: np.ndarray, levels: np.ndarray) -> np.ndarray:
    med = q[..., len(levels) // 2]
    return np.abs(med - truth).mean(axis=1)


def mase(q: np.ndarray, truth: np.ndarray, ctx: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """MASE with the in-sample naive (last value) scale of each stroke's context."""
    med = q[..., len(levels) // 2]
    naive = np.abs(np.diff(ctx, axis=1)).mean(axis=1)
    return np.abs(med - truth).mean(axis=1) / np.maximum(naive, 1e-8)


def coverage90(q: np.ndarray, truth: np.ndarray) -> np.ndarray:
    lo, hi = q[..., 1], q[..., -2]
    return ((truth >= lo) & (truth <= hi)).mean(axis=1)


def summarize(q: np.ndarray, truth: np.ndarray, ctx: np.ndarray, levels: np.ndarray) -> dict:
    med = q[..., len(levels) // 2]
    naive = np.abs(np.diff(ctx, axis=1)).mean(axis=1)
    per_stroke_abs = np.abs(med - truth).mean(axis=1)
    return {
        "pinball": float(pinball(q, truth, levels).mean()),
        "mae": float(per_stroke_abs.mean()),
        "mase": float((per_stroke_abs / np.maximum(naive, 1e-8)).mean()),
        "cov90": float(coverage90(q, truth).mean()),
        "beats_naive": float(np.mean(per_stroke_abs < np.abs(truth - ctx[:, -1:]).mean(axis=1))),
    }


def by_kind(q: np.ndarray, truth: np.ndarray, ctx: np.ndarray, levels: np.ndarray) -> list[dict]:
    """The corpus cycles four stroke families (`s % 4`), so group by row index."""
    rows = []
    for kind, name in enumerate(KIND_NAMES):
        idx = np.arange(kind, len(q), len(KIND_NAMES))
        if len(idx) == 0:
            continue
        s = summarize(q[idx], truth[idx], ctx[idx], levels)
        rows.append({"kind": name, "strokes": len(idx), **s})
    return rows
