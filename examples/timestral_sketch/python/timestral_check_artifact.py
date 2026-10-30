"""Verify the packed ternary artifact against PyTorch, weight by weight.

The artifact is the deployable thing: trits per output row, one scale per row, plus the
fp32 norm/bias/query tensors. This script parses it, rebuilds the deployed fp32 weights
(`scale * trit`), loads them into the reference model, and forecasts the same corpus the
Zig engine scored. If the two agree, the packer, the unpacker and the forward pass are all
consistent; if they disagree, the difference is in one of those, not in the training.

    python examples/timestral_sketch/python/timestral_check_artifact.py examples/timestral_sketch/modeldata/timestral_packed.bin
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch

EX = Path(__file__).resolve().parents[1]   # examples/timestral_sketch
ROOT = EX.parents[1]                          # repository root
sys.path.insert(0, str(EX / "reference"))

from timestral import TimestralForecaster  # noqa: E402
from timestral.config import TimestralConfig  # noqa: E402
from timestral.model import TimestralModel  # noqa: E402

from timestral_metrics import summarize  # noqa: E402

CTX, H = 256, 96


def packed_len(n: int) -> int:
    return (n + 4) // 5


def unpack5(dst: np.ndarray, src: bytes) -> None:
    """Mirror of framework/ternary.zig unpack5: peel digits off the top of each byte."""
    i = 0
    for g in range(len(src)):
        b = int(src[g])
        for _ in range(5):
            if i >= len(dst):
                return
            b *= 3
            dst[i] = (b >> 8) - 1
            b &= 0xFF
            i += 1


def read_artifact(path: Path) -> tuple[dict, list[tuple]]:
    """Parse the container into config values and a list of (kind, payload) records.

    The record stream is positional: a matrix record carries a leading kind byte, an fp32
    vector does not, and the canonical order (patch block, query, layers, out-norm, head)
    is what says which is which — exactly what the Zig encoder writes.
    """
    data = path.read_bytes()
    assert data[:4] == b"TRTL", "not a Timestral artifact"
    pos = 4
    version, = struct.unpack_from("<I", data, pos)
    pos += 4
    assert version == 2, version
    keys = ["patch_size", "d_model", "num_heads", "mlp_hidden", "n_layers", "n_knots", "block_size", "rope_dim"]
    cfg = {}
    for k in keys:
        cfg[k], = struct.unpack_from("<I", data, pos)
        pos += 4
    for k in ["width_factor", "norm_eps", "rope_theta", "min_scale", "width_scale"]:
        cfg[k], = struct.unpack_from("<f", data, pos)
        pos += 4
    cfg["use_asinh"] = bool(data[pos])
    cfg["attend_missing"] = bool(data[pos + 1])
    pos += 2
    cfg["schedule"] = list(data[pos : pos + cfg["n_layers"]])
    pos += cfg["n_layers"]
    cfg["knots"] = list(struct.unpack_from(f"<{cfg['n_knots']}f", data, pos))
    pos += 4 * cfg["n_knots"]

    order = ["mat", "vec", "mat", "vec", "mat", "vec", "vec"]
    axes_per_kind = {0: 1, 1: 1, 2: 2, 3: 2}
    for kind in cfg["schedule"]:
        for _ in range(axes_per_kind[kind]):
            order += ["mat", "mat", "mat", "vec"]
        order += ["mat", "mat", "vec"]
    order += ["vec", "mat", "mat", "mat"]

    records = []
    for want in order:
        if want == "vec":
            n, = struct.unpack_from("<I", data, pos)
            pos += 4
            vals = np.frombuffer(data, dtype="<f4", count=n, offset=pos).copy()
            pos += 4 * n
            records.append(("vec", vals))
        else:
            kind = data[pos]
            pos += 1
            assert kind == 1, f"expected a ternary matrix record, got kind {kind} at {pos - 1}"
            rows, cols = struct.unpack_from("<II", data, pos)
            pos += 8
            scales = np.frombuffer(data, dtype="<f4", count=rows, offset=pos).copy()
            pos += 4 * rows
            nbytes, = struct.unpack_from("<I", data, pos)
            pos += 4
            trits = np.empty(rows * cols, dtype=np.int8)
            unpack5(trits, data[pos : pos + nbytes])
            pos += nbytes
            records.append(("mat", trits.reshape(rows, cols).astype(np.float32) * scales[:, None]))
    assert pos == len(data), f"{len(data) - pos} trailing bytes"
    return cfg, records


def build_model(cfg: dict, records: list) -> TimestralModel:
    c = TimestralConfig(
        patch_size=cfg["patch_size"], d_model=cfg["d_model"], num_heads=cfg["num_heads"],
        mlp_hidden_dim=cfg["mlp_hidden"],
        block_schedule=tuple({0: "time", 1: "variate", 2: "time_variate", 3: "variate_time"}[k] for k in cfg["schedule"]),
        # the artifact stores fp32 knots; round so the symmetric-averaging assert holds
        quantile_knots=tuple(round(k, 6) for k in cfg["knots"]),
        block_size=cfg["block_size"],
        context_length=32768,
        rope_theta=cfg["rope_theta"], norm_eps=cfg["norm_eps"], minimum_scale=cfg["min_scale"],
        use_asinh=cfg["use_asinh"], attend_missing_patches=cfg["attend_missing"],
        quantile_width_factor=cfg["width_factor"] * cfg["width_scale"],
    )
    model = TimestralModel(c)
    it = iter(records)
    state = {}

    def take(kind):
        k, v = next(it)
        assert k == kind, (k, kind)
        return v

    state["representation.patch_proj.hidden_layer.weight"] = take("mat")
    state["representation.patch_proj.hidden_layer.bias"] = take("vec")
    state["representation.patch_proj.output_layer.weight"] = take("mat")
    state["representation.patch_proj.output_layer.bias"] = take("vec")
    state["representation.patch_proj.residual_layer.weight"] = take("mat")
    state["representation.patch_proj.residual_layer.bias"] = take("vec")
    state["representation.forecast_query"] = take("vec")
    for i in range(cfg["n_layers"]):
        axes = {0: (True,), 1: (False,), 2: (True, False), 3: (False, True)}[cfg["schedule"][i]]
        for j in range(len(axes)):
            state[f"transformer.layers.{i}.attns.{j}.wQKV.weight"] = take("mat")
            state[f"transformer.layers.{i}.attns.{j}.wO.weight"] = take("mat")
            state[f"transformer.layers.{i}.attns.{j}._gate.weight"] = take("mat")
            state[f"transformer.layers.{i}.attn_norms.{j}.scale"] = take("vec")
        state[f"transformer.layers.{i}.mlp.w12.weight"] = take("mat")
        state[f"transformer.layers.{i}.mlp.w3.weight"] = take("mat")
        state[f"transformer.layers.{i}.mlp_norm.scale"] = take("vec")
    state["transformer.out_norm.scale"] = take("vec")
    state["head.projection.hidden_layer.weight"] = take("mat")
    state["head.projection.output_layer.weight"] = take("mat")
    state["head.projection.residual_layer.weight"] = take("mat")
    missing, unexpected = model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=False)
    assert not unexpected, unexpected
    return model


def main() -> None:
    art = Path(sys.argv[1]) if len(sys.argv) > 1 else EX / "modeldata" / "timestral_packed.bin"
    zig_path = EX / "modeldata" / "timestral_forecast_ternary.bin"
    corpus = np.fromfile(EX / "modeldata" / "timestral_strokes.bin", dtype=np.float32).reshape(-1, CTX + H)
    ctx, truth = corpus[:, :CTX], corpus[:, CTX:]

    cfg, records = read_artifact(art)
    print(f"artifact {art.name}: {len(records)} records, d_model {cfg['d_model']}, layers {cfg['n_layers']}")
    model = build_model(cfg, records)
    print(f"rebuilt params {sum(p.numel() for p in model.parameters()):,}")

    fc = TimestralForecaster(model)
    levels = np.asarray(fc.quantile_levels, dtype=np.float32)
    with torch.inference_mode():
        q = fc.forecast(torch.from_numpy(ctx), prediction_length=H, clip_nonnegative=False).quantiles[:, 0].numpy()
    print(f"torch-on-dequantized {summarize(q, truth, ctx, levels)}")

    if zig_path.exists():
        q_zig = np.fromfile(zig_path, dtype=np.float32).reshape(len(corpus), H, len(levels))
        diff = np.abs(q - q_zig)
        print(f"max |torch(dequantized) - zig| {diff.max():.6f} on magnitude {np.max(np.abs(q)):.3f}")
        print(f"relative {diff.max() / np.max(np.abs(q)):.2e}")
        ok = diff.max() / np.max(np.abs(q)) < 1e-3
        print("packed artifact parity:", "OK" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
