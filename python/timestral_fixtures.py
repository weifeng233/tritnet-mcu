"""Reference fixtures for the Zig Timestral engine.

Runs the vendored PyTorch reference on fixed strokes and dumps every intermediate tensor
of one forecast pass to a flat binary the Zig tests read back:

    u32 magic 'TRFX' | u32 version | u32 count
    per tensor: u32 name_len | name | u32 ndim | u32 dims[ndim] | f32 data (little endian)

Three cases are covered:

    block      one decoder block (8 context patches, 8 horizon patches), internals
    forecast   the forecaster path on h=96 (refine + cap, no mirror)
    sym        the same with symmetric averaging, i.e. what the sketch app requests

    python python/timestral_fixtures.py
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))

from timestral import TimestralForecaster  # noqa: E402
from timestral.model import TimestralModel  # noqa: E402
from timestral.masks import (  # noqa: E402
    build_forecast_positions,
    build_forecast_time_mask,
    build_two_stream_variate_mask,
    compute_enc_key_validity,
    group_first_observed_patch,
)
from timestral.scaler import apply_affine_asinh_masked, causal_patch_loc_scale  # noqa: E402

MODEL_DIR = ROOT / "modeldata" / "timestral-5m"
OUT = ROOT / "modeldata" / "fixtures" / "timestral_fixture.bin"

CTX = 256  # 8 patches
H = 96


def strokes(r: int, t: int) -> tuple[np.ndarray, np.ndarray]:
    """`r` strokes of `t` steps with a leading missing run on the first row."""
    rng = np.random.default_rng(11)
    x = np.empty((r, t), dtype=np.float32)
    tt = np.arange(t)
    for i in range(r):
        if i == 0:
            x[i] = 0.6 * np.sin(2 * np.pi * tt / 64.0) + 0.02 * tt / t
        elif i == 1:
            x[i] = np.cumsum(rng.normal(0, 0.08, t))
        else:
            x[i] = np.repeat(rng.uniform(-1, 2, (t + 31) // 32), 32)[:t] + 3.5
    mask = np.ones_like(x, dtype=bool)
    mask[0, : 2 * 32] = False  # two leading missing patches: left padding
    x = np.where(mask, x, 0.0)
    return x, mask


class Dump:
    def __init__(self) -> None:
        self.items: list[tuple[str, np.ndarray]] = []

    def add(self, name: str, t) -> None:
        if isinstance(t, torch.Tensor):
            a = t.detach().cpu().numpy()
        else:
            a = np.asarray(t)
        if a.dtype == np.bool_:
            a = a.astype(np.float32)
        else:
            a = a.astype(np.float32)
        self.items.append((name, a))

    def write(self, path: Path) -> None:
        with path.open("wb") as f:
            f.write(struct.pack("<4sII", b"TRFX", 1, len(self.items)))
            for name, a in self.items:
                b = name.encode()
                f.write(struct.pack("<I", len(b)))
                f.write(b)
                f.write(struct.pack("<I", a.ndim))
                f.write(struct.pack("<" + "I" * a.ndim, *a.shape))
                f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())


def capture_block(model: TimestralModel, inputs: np.ndarray, observed: np.ndarray, h_patches: int) -> Dump:
    """Replica of `TimestralModel._predict_context_only` that records every intermediate."""
    d = Dump()
    c = model.config
    ps = c.patch_size
    inp = torch.from_numpy(inputs)
    obs = torch.from_numpy(observed)
    rows, t = inp.shape
    ll = t // ps
    group_ids = torch.arange(rows)

    d.add("inputs", inp)
    d.add("observed", obs)
    scaled, loc, scale = model.scale(inp, obs)
    d.add("scaled", scaled)
    d.add("loc", loc)
    d.add("scale", scale)

    x_enc = model.representation(scaled, obs)
    x_dec = model.representation.forecast_tokens(rows, h_patches)
    d.add("x_enc", x_enc)
    d.add("x_dec", x_dec)

    obs_per_patch = obs.unflatten(-1, (ll, ps)).any(dim=-1)
    key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
    time_mask = build_forecast_time_mask(ll, h_patches, key_valid, inp.device)
    variate_mask = build_two_stream_variate_mask(group_ids, key_valid, h_patches)
    _, inverse = torch.unique(group_ids, return_inverse=True)
    first_obs = group_first_observed_patch(obs_per_patch, inverse, rows)[inverse]
    positions = build_forecast_positions(ll, h_patches, first_obs)
    d.add("obs_per_patch", obs_per_patch)
    d.add("key_valid", key_valid)
    d.add("time_mask", time_mask[:, 0])
    d.add("positions", positions)
    d.add("first_obs", first_obs)

    x = torch.cat([x_enc, x_dec], dim=1)
    d.add("x_in", x)
    cos, sin = model.transformer.rope(x, positions)
    d.add("rope_cos", cos)
    d.add("rope_sin", sin)
    for i, layer in enumerate(model.transformer.layers):
        x = layer(x, time_mask, variate_mask, cos, sin)
        d.add(f"layer{i}.out", x)
    h_out = model.transformer.out_norm(x)
    d.add("out_norm", h_out)
    preds = model.head(h_out[:, ll:, :])
    d.add("head_out", preds)
    anchor_loc = loc.unflatten(-1, (ll, ps))[..., 0][:, ll - 1 : ll]
    anchor_scale = scale.unflatten(-1, (ll, ps))[..., 0][:, ll - 1 : ll]
    from timestral.scaler import denormalize_quantiles

    d.add("anchor_loc", anchor_loc)
    d.add("anchor_scale", anchor_scale)
    final = denormalize_quantiles(preds, anchor_loc.unsqueeze(-1), anchor_scale.unsqueeze(-1), c.use_asinh)
    d.add("block_quantiles", final)
    return d


def main() -> None:
    torch.manual_seed(0)
    x, m = strokes(3, CTX)
    fc = TimestralForecaster.from_pretrained(MODEL_DIR)
    fc.model.eval()
    d = capture_block(fc.model, x, m, 8)

    with torch.inference_mode():
        flat = fc.forecast(torch.from_numpy(x), prediction_length=H, clip_nonnegative=False,
                           symmetric_averaging=False).quantiles.numpy()
        sym = fc.forecast(torch.from_numpy(x), prediction_length=H, clip_nonnegative=False,
                          symmetric_averaging=True).quantiles.numpy()
    d.add("forecast", flat)
    d.add("forecast_sym", sym)
    d.add("quantile_levels", np.asarray(fc.quantile_levels))
    d.add("width_factor", np.asarray([fc.config.quantile_width_factor]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.write(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(d.items)} tensors)")
    print(json.dumps({n: list(a.shape) for n, a in d.items}, indent=0)[:600])


if __name__ == "__main__":
    main()
