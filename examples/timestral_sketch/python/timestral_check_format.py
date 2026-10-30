"""Parse a TRTL artifact using only docs/ARTIFACT-FORMAT.md as the specification.

This is a deliberately separate implementation from
examples/timestral_sketch/timestral/artifact.zig: it
exists so that the format document can be checked rather than trusted. If the document
and the writer disagree, this script desynchronizes and the byte count at the end is
wrong -- which is exactly how the "vector records carry a kind byte" error was found
(the doc said `u8 0 | u32 n | f32 data`, the writer has never emitted that byte).

    python examples/timestral_sketch/python/timestral_check_format.py examples/timestral_sketch/modeldata/timestral_packed.bin

Exits 0 when the file is consumed exactly, with a summary of what it contained.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

# schedule kind -> number of attention axes (ARTIFACT-FORMAT.md section 2)
AXES = {0: 1, 1: 1, 2: 2, 3: 2}


class Cursor:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError(f"desync: wanted {n} bytes at {self.pos}, only {len(self.data) - self.pos} left")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def u8(self) -> int:
        return self.take(1)[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.take(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.take(8))[0]


def read_vector(r: Cursor) -> int:
    """u32 n | f32 values[n] -- no record tag."""
    n = r.u32()
    for _ in range(n):
        r.f32()
    return n


def read_matrix(r: Cursor) -> tuple[str, int, int]:
    """u8 kind | (dense: rows, cols, f32 data) | (ternary: rows, cols, scales, nbytes, trits)."""
    kind = r.u8()
    if kind == 0:
        rows, cols = r.u32(), r.u32()
        for _ in range(rows * cols):
            r.f32()
        return "dense", rows, cols
    if kind != 1:
        raise ValueError(f"unknown record kind {kind} at {r.pos - 1}")
    rows, cols = r.u32(), r.u32()
    for _ in range(rows):
        r.f32()
    nbytes = r.u32()
    if nbytes != (rows * cols + 4) // 5:
        raise ValueError(f"record at {r.pos} claims {nbytes} packed bytes for {rows}x{cols}")
    r.take(nbytes)
    return "ternary", rows, cols


def parse(path: Path) -> None:
    r = Cursor(path.read_bytes())

    magic = r.take(4)
    if magic != b"TRTL":
        raise ValueError(f"not a TRTL file: {magic!r}")
    version = r.u32()
    if version != 2:
        raise ValueError(f"unsupported version {version}")
    patch_size, d_model, num_heads, mlp_hidden = r.u32(), r.u32(), r.u32(), r.u32()
    n_layers, n_knots, block_size, rope_dim = r.u32(), r.u32(), r.u32(), r.u32()
    width_factor, norm_eps, rope_theta, min_scale, width_scale = (r.f32() for _ in range(5))
    use_asinh, attend_missing = r.u8(), r.u8()
    schedule = [r.u8() for _ in range(n_layers)]
    knots = [r.f32() for _ in range(n_knots)]

    print(f"header    {path.name}: TRTL v{version}, {d_model}d {num_heads}h {n_layers}L, "
          f"patch {patch_size}, {n_knots} knots, rope {rope_dim}")
    print(f"          asinh={bool(use_asinh)} attend_missing={bool(attend_missing)} "
          f"width_scale={width_scale:.4f} (width_factor {width_factor}, eps {norm_eps:g}, theta {rope_theta:g})")
    print(f"schedule  {schedule}  knots {knots[0]:.2f}..{knots[-1]:.2f}")

    records: list[str] = []

    def vector(label: str) -> None:
        n = read_vector(r)
        records.append(f"vector  {label:28} n={n}")

    def matrix(label: str) -> None:
        kind, rows, cols = read_matrix(r)
        records.append(f"{kind:7} {label:28} {rows}x{cols}")

    # 1. patch block, 2. forecast query
    for name in ("patch.hidden", "patch.hidden_b", "patch.out", "patch.out_b", "patch.res", "patch.res_b"):
        (vector if name.endswith("_b") else matrix)(name)
    vector("forecast_query")

    # 3. layers, in the order the schedule says
    for i, kind in enumerate(schedule):
        for j in range(AXES.setdefault(kind, 0)):
            matrix(f"layers[{i}].attn[{j}].wqkv")
            matrix(f"layers[{i}].attn[{j}].wO")
            matrix(f"layers[{i}].attn[{j}].gate")
            vector(f"layers[{i}].attn_norm[{j}]")
        matrix(f"layers[{i}].w12")
        matrix(f"layers[{i}].w3")
        vector(f"layers[{i}].mlp_norm")

    # 4. out norm, 5. head
    vector("out_norm")
    for name in ("head.hidden", "head.out", "head.res"):
        matrix(name)

    if r.pos != len(r.data):
        raise ValueError(f"trailing bytes: consumed {r.pos} of {len(r.data)}")
    print(f"records   {len(records)} records, {r.pos} bytes consumed == file size")
    for line in records:
        print("  " + line)


def main() -> int:
    args = sys.argv[1:]
    if not args:
        args = ["examples/timestral_sketch/modeldata/timestral_packed.bin"]
    for a in args:
        p = Path(a)
        if not p.exists():
            print(f"missing   {p} (rebuild with `make quantize-timestral`)")
            continue
        try:
            parse(p)
        except ValueError as e:
            print(f"FAIL      {p}: {e}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
