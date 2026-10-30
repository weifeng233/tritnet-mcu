# Artifact formats

- **Applies to:** every `zig build` artifact in this repository, Zig 0.16.0, `main` as of v0.2.0
- **Last verified:** 2026-10-30
- **Verification:** `zig build test --summary all` (49 tests, including the golden
  vectors below), `make deploy-check`, `make timestral-check`

There are five binary formats in this repository plus two raw dumps. They share one
primitive — the ternary codec — and differ only in what bookkeeping surrounds it.
This document specifies all of them well enough to write a reader in another language,
which is the point: the C engines in `c/` are exactly such readers.

| magic | first bytes | written by | read by |
| --- | --- | --- | --- |
| `TRTL` v2 | `54 52 54 4C 02 00 00 00` | `examples/timestral_sketch/timestral/artifact.zig` | `timestral-cli`, the parity tools |
| `MGPTQ001` | `4D 47 50 54 51 30 30 31` | `examples/gpt_names/gpt_names.zig` | same example, round-trip check |
| `CNN1Q002` | `43 4E 4E 31 51 30 30 32` | `examples/gesture_cnn/gesture_cnn.zig` | `examples/gesture_cnn/c/cnn1d_inference.c` |
| `CNN1V001` | `43 4E 4E 31 56 30 30 31` | `examples/gesture_cnn/gesture_cnn.zig` | `examples/gesture_cnn/c/cnn1d_inference.c` |
| `NCPSQ001` | `4E 43 50 53 51 30 30 31` | `examples/liquid_sine/liquid_sine.zig` | `examples/liquid_sine/c/ncps_inference.c` |
| `NCPSV001` | `4E 43 50 53 56 30 30 31` | `examples/liquid_sine/liquid_sine.zig` | `examples/liquid_sine/c/ncps_inference.c` |
| `TRFX` v1 | `54 52 46 58 01 00 00 00` | `examples/timestral_sketch/python/timestral_fixtures.py` | `examples/timestral_sketch/timestral/fixture.zig` |

All integers are **little-endian**. All `f32` are IEEE-754 binary32, stored as their
bit pattern in a little-endian `u32`. Nothing is memory-mapped or native endian: a
file written on x86 Windows reads identically on ARM Linux.

---

## 1. The shared primitive: 5 trits per byte

Weights are ternary, `t ∈ {-1, 0, +1}`. Five trits fit in one byte because
`3^5 = 243 ≤ 256`, giving **1.6 bits per weight** against the `log2(3) ≈ 1.585`
information-theoretic floor. Scheme and rationale: <https://compilade.net/blog/ternary-packing>.

### Encode five trits into one byte

```
b = 0
for t in [t0, t1, t2, t3, t4]:        # most significant trit first
    b = b * 3 + (t + 1)               # b is now in 0..242
byte = ceil(b * 256 / 243)            # integer: (b * 256 + 242) / 243
```

The ceiling rescale matters. A plain `b` stored in a byte would truncate the top
trit when reading it back out; rounding up cancels that truncation, and stays
injective over the 243 distinct blocks (proved by the test
`distinct blocks pack to distinct bytes`).

### Decode one byte into five trits

```
b = byte
for i in 0..4:
    b = b * 3
    trit = (b >> 8) - 1               # top bits of the 10-bit product
    b = b & 0xFF                      # keep the remainder
```

No division and no modulo in the inner loop: one multiply, one shift, one mask. That
is what makes the C engines cheap on a Cortex-M.

### Length and padding

`packed_len(n) = ceil(n / 5)`. A matrix with `rows * cols` trits that is not a multiple
of five is zero-padded on the right before packing; the reader knows `n` from the
header, so padding is discarded on decode and never leaks into a weight.

### Golden vectors

If your encoder produces these bytes for these trits, it agrees with this repository's:

| trits (t0..t4) | base-3 value `b` | byte | hex |
| --- | --- | --- | --- |
| `[+1,  0, -1,  0, +1]` | 194 | 205 | `0xCD` |
| `[-1, -1, -1, -1, -1]` | 0 | 0 | `0x00` |
| `[+1, +1, +1, +1, +1]` | 242 | 255 | `0xFF` |
| `[+1, +1, +1, +1, -1]` | 240 | 253 | `0xFD` |

Worked example for the first row: digits are `[2, 1, 0, 1, 2]`, so
`b = ((((2*3 + 1)*3 + 0)*3 + 1)*3 + 2) = 194`; then
`ceil(194 * 256 / 243) = ceil(204.38) = 205`. Decoding `205`: `205*3 = 615`,
`615 >> 8 = 2` → trit `+1`, `615 & 0xFF = 103`; `103*3 = 309`, `309 >> 8 = 1` → trit
`0`, and so on.

These four vectors are asserted by
[`src/framework/ternary.zig`](../src/framework/ternary.zig) in the test
`golden vectors: the worked examples in docs/ARTIFACT-FORMAT.md`, so the document and
the code cannot drift apart silently.

### Version policy for the codec

The packing is frozen. Changing the block size, the digit order, or dropping the
ceiling rescale is a **format break**: every container in this repository must bump
its version, and every reader is expected to reject the old files rather than guess.

---

## 2. `TRTL` v2 — the Timestral deployment artifact

One file carries everything `timestral-cli` needs: architecture config, quantile
knots, fp32 tensors, and every projection as trits plus one absmean scale per output
row. The deployed weight is `scale[row] * trit`, and nothing else is stored per
matrix. 21 MB of fp32 becomes **1.15 MB**.

### Header

| field | type | value in Timestral-5M | meaning |
| --- | --- | --- | --- |
| magic | `u8[4]` | `TRTL` | fixed |
| version | `u32` | 2 | incompatible changes must increment |
| patch_size | `u32` | 32 | time steps per patch |
| d_model | `u32` | 192 | residual width |
| num_heads | `u32` | 3 | attention heads per axis |
| mlp_hidden | `u32` | 512 | SwiGLU hidden width |
| n_layers | `u32` | 8 | layer count, `≤ 16` |
| n_knots | `u32` | 21 | quantiles per step, `≤ 32` |
| block_size | `u32` | 32 | decode block (patches) |
| rope_dim | `u32` | 32 | RoPE dimensions per head |
| width_factor | `f32` | 3.0 | quantile width multiplier |
| norm_eps | `f32` | 1e-6 | RMSNorm epsilon |
| rope_theta | `f32` | 10000.0 | RoPE base |
| min_scale | `f32` | 1e-6 | floor for patch scaling |
| width_scale | `f32` | calibrated | post-quantization width calibration |
| use_asinh | `u8` | 1 | bool: asinh/sinh patch transform |
| attend_missing | `u8` | 1 | bool: attend over unobserved patches |
| schedule[n_layers] | `u8[n]` | — | per-layer attention axis schedule, see below |
| knots[n_knots] | `f32[n]` | 0.01 … 0.99 | quantile levels |

`schedule` values are `0` = time, `1` = variate, `2` = time then variate,
`3` = variate then time. Reading a schedule entry outside 0..3 is an error, not a
default.

`width_scale` is the one field that is *not* a property of the checkpoint: it is fitted
after quantization on 96 synthetic strokes (`timestral-cli quantize` does it unless
`--no-calibrate` is passed, which is how the plain-PTQ baseline is produced). Two
artifacts built from the same weights with different calibration therefore differ only
in this `f32`.

### Tensor records

After the header come parameter records in a **fixed canonical order**. There is no
table of contents and no name strings: the order *is* the schema, so a reader that
skips a record is broken rather than lenient.

Matrix records carry a one-byte kind tag; **vector records do not**. That asymmetry is
deliberate: a matrix has to say whether it is packed or dense, a vector never has a
choice to make.

```
fp32 vector:      u32 n | f32 values[n]
dense matrix:     u8 0 | u32 rows | u32 cols | f32 values[rows*cols]
ternary matrix:   u8 1 | u32 rows | u32 cols | f32 scales[rows] | u32 nbytes | u8 trits[nbytes]
```

`nbytes` must equal `ceil(rows * cols / 5)`; a mismatch is `error.BadRecord`. Row `r`
of the matrix owns `scales[r]` and the `cols` trits that follow it in row-major order.

> A reader written from an earlier revision of this document expected a `u8 0` in front
> of every vector and desynchronized at the first vector in the file. The asymmetry
> above is the corrected rule, and the test
> `record layout round-trips a small model without a checkpoint` executes this exact
> layout — vectors, a dense matrix and ternary matrices — so the document and the code
> cannot drift apart again.

Canonical order:

1. patch block: `matrix`, `vector`, `matrix`, `vector`, `matrix`, `vector`
   (hidden projection, its bias, output projection, its bias, residual, its bias)
2. `forecast_query` — `vector`
3. for each layer, in the order of its schedule axis roles (one axis for `time` or
   `variate`, two for the mixed kinds): `wqkv`, `wO`, `gate`, and an attention-norm
   `vector` per axis; then the MLP (`w12` matrix, `w3` matrix, `mlp_norm` vector)
4. `out_norm` — `vector`
5. head: `hidden` matrix, `out` matrix, `res` matrix

`timestral-cli eval` and `serve` read only this file: no safetensors parser, no
PyTorch, no Python.

### Incompatibility

```zig
if (!std.mem.eql(u8, magic, MAGIC)) return error.BadMagic;
if (try r.u32v() != VERSION) return error.BadVersion;
```

A `TRTL` file with a different version is rejected. We do not attempt to read v1
files with the v2 reader: a wrong forecast is worse than a clear error. Same rule for
`BadConfig` (`n_layers` or `n_knots` beyond the in-memory maximum), `BadRecord`
(unknown record kind, wrong `nbytes`) and `TrailingBytes` (the file ended later than
the canonical order says it should — a reader that ignores the tail is hiding a
disagreement about the format).

---

## 3. `MGPTQ001`, `CNN1Q002`, `NCPSQ001` — the framework's example containers

These three are the same pattern: an 8-byte ASCII magic (the version is *inside* the
magic: `Q001`), a run of `u32` dimensions, then records. They are consumed by the C
engines in `c/` and by each example's own round-trip test.

The shared ternary record inside them is:

```
f32 absmean_scale | u32 nbytes | u8 trits[nbytes]
```

— one scale for the whole matrix, not per row (these models are small enough that one
absmean is what the trainer used; see `framework/quant.zig`).

| format | header after magic | body |
| --- | --- | --- |
| `MGPTQ001` | `u32 n_embd, n_head, n_layer, block_size, mlp_dim, vocab_size, BOS, n_uchars`, then `u8 uchars[n_uchars]` (the tokenizer) | `f32 wte[vocab_size*n_embd]`, `f32 wpe[block_size*n_embd]`, then per layer `matrix` × 4 (wq, wk, wv, wo) and `matrix` × 2 (fc1, fc2), then `matrix` lm_head. Norms are recomputed, not stored; embeddings stay fp32. |
| `CNN1Q002` | `u32 C0, T0, C1, K1, C2, K2, n_class` | `matrix` × 3 (conv1, conv2, fc), then the fp32 bias arrays `C1`, `C2`, `n_class`, then `f64 expected_accuracy` — the trainer's val accuracy, which the C engine recomputes and compares |
| `NCPSQ001` | `u32 in_size, units_total, out_dim, n_layers, mode, bb_units, proj_size, cell_kind` | if `cell_kind` = 1: `matrix` × 2 (synapse, sensory) then 13 fp32 physical parameter arrays (`sigma, mu, erev, sensory_sigma, sensory_mu, sensory_erev, gleak, vleak, cm, input_w, input_b, output_w, output_b`); then an optional readout; then per layer `u32 n_sens, u32 n_out` and `matrix + f32 bias` records for `ff1`, and either (`ff2`, `t_a`, `t_b` matrices) or (`tau`, `a_par` vectors); then an optional projection; finally `f64 expected_mse` used by the round-trip check |

The `V001` files are their held-out evaluation sets, so the C engine can score without
re-deriving the dataset:

| format | layout |
| --- | --- |
| `CNN1V001` | `u32 n_val, C0, T0, n_class` | `f32 windows[n_val][C0][T0]`, then `u8 labels[n_val]` |
| `NCPSV001` | `u32 n_val, T_seq, in_dim, out_dim` | `f32 x[n_val][T_seq][in_dim]`, then `f32 y[n_val][T_seq][out_dim]` |

A `V001` file is not a model and cannot be loaded as one: the magic differs on purpose.

---

## 4. `TRFX` v1 — reference fixtures

Written by `examples/timestral_sketch/python/timestral_fixtures.py`, read by `examples/timestral_sketch/timestral/fixture.zig`. It is a
flat named-tensor dump of one PyTorch forward pass, so the Zig port can be compared
stage by stage in a clone that has no checkpoint.

```
u32 magic "TRFX" | u32 version (1) | u32 count
per tensor: u32 name_len | u8 name[name_len] | u32 ndim | u32 dims[ndim] | f32 data[prod(dims)]
```

No alignment padding, no trailing checksum. Names are the reference implementation's
own stage labels (`inputs`, `block_quantiles`, `forecast_sym`, …), which is why the
fixtures must be generated from `examples/timestral_sketch/reference/timestral` and not from this port: a reader
agreeing with an echo of itself proves nothing.

---

## 5. Raw dumps (headerless)

Two files carry no header because they are inputs to a comparison, not artifacts:

| file | shape | produced by |
| --- | --- | --- |
| `examples/timestral_sketch/modeldata/timestral_strokes.bin` | `f32[24][ctx + h]`, row-major, `ctx=256`, `h=96` | `timestral-cli eval --dump` |
| `examples/timestral_sketch/modeldata/timestral_forecast_*.bin` | `f32[n][h][n_knots]` | `timestral-cli forecast --out` |

Both are little-endian `f32` and are documented as much by
`examples/timestral_sketch/python/timestral_parity.py` as by this file: the parity script reads them with
`numpy.frombuffer(..., dtype="<f4")`.

---

## 6. Reproducing the artifacts, and checking this document

```bash
zig build test --summary all            # includes the codec golden vectors and the
                                        # record-layout round-trip
make quantize-timestral                 # TRTL from examples/timestral_sketch/modeldata/timestral_qat
make quantize-timestral-ptq             # TRTL baseline, --no-calibrate
make deploy-check                       # CNN1Q002 / NCPSQ001 through the C engines
make timestral-check                    # TRTL vs PyTorch, end to end
python examples/timestral_sketch/python/timestral_check_format.py examples/timestral_sketch/modeldata/timestral_packed.bin
```

That last command is the point of this page made executable:
`examples/timestral_sketch/python/timestral_check_format.py` is a reader written from **this document alone**,
not from the Zig code. It walks the canonical order, refuses anything it does not
recognize, and prints the consumed byte count — which has to equal the file size. When
this document claimed vectors carried a kind byte, that script desynchronized at the
first vector; it is here so the next such mistake is caught by a command instead of by
a reader with a debugger.

Sizes quoted above (21 MB → 1.15 MB) are from `examples/timestral_sketch/modeldata/timestral_eval_*.json`,
which records the byte count it measured next to the metrics. On the released
`timestral_packed.bin` — a release asset rather than a committed file, since 2.3 MB of
binaries do not belong in git history — the checker reports 75 records and 1,152,070
bytes consumed.
