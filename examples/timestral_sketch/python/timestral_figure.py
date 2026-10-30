"""Figure for docs/TIMESTRAL.md: the same stroke forecast by fp32 and by the ternary model.

Reads the stroke corpus written by `timestral-cli eval --dump` plus two raw forecast dumps
(`timestral-cli forecast --out ...`, one per backend), and draws the context, the truth,
and each model's median with its 5-95% and 25-75% bands.

    python examples/timestral_sketch/python/timestral_figure.py examples/timestral_sketch/modeldata/timestral_forecast_fp32.bin \
        examples/timestral_sketch/modeldata/timestral_forecast_ternary.bin docs/timestral-sketch.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from timestral_metrics import summarize

EX = Path(__file__).resolve().parents[1]   # examples/timestral_sketch
ROOT = EX.parents[1]                          # repository root
CTX, H = 256, 96
LEVELS = np.asarray([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])

BG = "#0d1117"
FG = "#c9d1d9"
MUTED = "#8b949e"
TERN = "#3fb950"
FP32 = "#8b949e"
TRUTH = "#d29922"


def band(ax, x, q, color, alpha, label=None):
    ax.fill_between(x, q[:, 1], q[:, -2], color=color, alpha=alpha * 0.5, lw=0, label=label)
    ax.fill_between(x, q[:, 4], q[:, -5], color=color, alpha=alpha, lw=0)


def main() -> None:
    corpus_path = EX / "modeldata" / "timestral_strokes.bin"
    fp32_path = Path(sys.argv[1]) if len(sys.argv) > 1 else EX / "modeldata" / "timestral_forecast_fp32.bin"
    tern_path = Path(sys.argv[2]) if len(sys.argv) > 2 else EX / "modeldata" / "timestral_forecast_ternary.bin"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "docs" / "timestral-sketch.png"

    corpus = np.fromfile(corpus_path, dtype=np.float32).reshape(-1, CTX + H)
    ctx, truth = corpus[:, :CTX], corpus[:, CTX:]
    q_fp32 = np.fromfile(fp32_path, dtype=np.float32).reshape(len(corpus), H, len(LEVELS))
    q_tern = np.fromfile(tern_path, dtype=np.float32).reshape(len(corpus), H, len(LEVELS))

    m_fp32 = summarize(q_fp32, truth, ctx, LEVELS)
    m_tern = summarize(q_tern, truth, ctx, LEVELS)

    fig = plt.figure(figsize=(11, 6.2), facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.35, 1], height_ratios=[1, 1], hspace=0.42, wspace=0.24)

    # two strokes, each the most typical of its family (closest to the family median pinball)
    per_stroke = np.abs(q_tern[..., 10] - truth).mean(axis=1)
    picks = []
    for kind, title in ((0, "drawn curve (typical)"), (1, "sinusoid (typical)")):
        idx = np.arange(kind, len(corpus), 5)
        pick = idx[np.argmin(np.abs(per_stroke[idx] - np.median(per_stroke[idx])))]
        picks.append((pick, title))
    for row, (idx, title) in enumerate(picks):
        ax = fig.add_subplot(gs[row, 0])
        ax.set_facecolor(BG)
        x_ctx = np.arange(CTX)
        x_fc = np.arange(CTX, CTX + H)
        ax.plot(x_ctx, ctx[idx], color=FG, lw=1.6, label="your stroke")
        ax.plot(x_fc, truth[idx], color=TRUTH, lw=1.2, ls=":", label="what happens")
        band(ax, x_fc, q_fp32[idx], FP32, 0.22)
        ax.plot(x_fc, q_fp32[idx][:, 10], color=FP32, lw=1.4, ls="--", label="fp32 median")
        band(ax, x_fc, q_tern[idx], TERN, 0.26)
        ax.plot(x_fc, q_tern[idx][:, 10], color=TERN, lw=1.8, label="ternary median")
        ax.axvline(CTX - 0.5, color=MUTED, lw=0.8, ls="-")
        ax.set_title(f"{title}: 21 quantiles, clampsym", color=MUTED, fontsize=9, loc="left")
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.grid(color="#21262d", lw=0.6)
        if row == 0:
            ax.legend(loc="lower left", fontsize=7.5, facecolor="#161b22", edgecolor="#30363d",
                      labelcolor=FG, framealpha=0.95, ncol=2)

    # metrics (horizontal bars on a log axis: pinball/mae are small, mase is not)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(BG)
    names = ["pinball", "mae", "mase", "cov90"]
    ypos = np.arange(len(names))[::-1]
    hgt = 0.34
    for name, off, color, metrics in (("fp32", hgt / 2, FP32, m_fp32), ("ternary (QAT)", -hgt / 2, TERN, m_tern)):
        vals = [max(metrics[n], 1e-3) for n in names]
        ax.barh(ypos + off, vals, hgt, color=color, label=name)
        for y, v in zip(ypos + off, vals):
            ax.text(v * 1.06, y, f"{v:.3g}", va="center", color=color, fontsize=7)
    ax.set_yticks(ypos, names, color=MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 40)
    ax.set_title("held-out strokes (24), log scale", color=MUTED, fontsize=9, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.grid(axis="x", color="#21262d", lw=0.6)
    ax.legend(fontsize=7.5, facecolor=BG, edgecolor=BG, labelcolor=FG, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

    # text panel
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(BG)
    ax.axis("off")
    lines = [
        ("artifact", "20.8 MB → 1.15 MB (18.1x)", FG),
        ("weights", "5.21 M params at 1.71 bits", FG),
        ("engine", "Zig, packed trits, no PyTorch", FG),
        ("parity vs torch", "3.9e-6 relative", FG),
        ("quantization", "absmean QAT + 5 trits/byte", FG),
    ]
    y = 0.92
    for key, val, color in lines:
        ax.text(0.0, y, key, color=MUTED, fontsize=8, transform=ax.transAxes)
        ax.text(0.42, y, val, color=color, fontsize=8, transform=ax.transAxes)
        y -= 0.16
    ax.set_title("the deploy leg", color=MUTED, fontsize=9, loc="left")

    fig.suptitle("Timestral-5M · sketch a series · fp32 vs ternary (1.7 bits/weight)",
                 color=FG, fontsize=12, x=0.012, ha="left", y=0.985)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"fp32 {m_fp32}")
    print(f"tern {m_tern}")


if __name__ == "__main__":
    main()
