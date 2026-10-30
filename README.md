# TritNet-MCU (tritnetmcu)

[![ci](https://github.com/weifeng233/tritnet-mcu/actions/workflows/ci.yml/badge.svg)](https://github.com/weifeng233/tritnet-mcu/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

English | [简体中文](README.zh-CN.md)

A pure-Zig train, quantize and deploy framework for ternary-weight {-1, 0, +1}
MCU models. It needs nothing beyond libc: one command runs fp32 pretraining,
BitNet-style quantization-aware training, held-out validation, a ternary-packed
model export, and a round-trip check that the packed file is lossless. Each example
takes seconds on a laptop.

The framework lives in `src/framework.zig` (RNG, ternary codec, absmean QAT
quantizer, Adam with a cosine schedule, the packed-model container, timing). The
`examples/` directory drives it with four model families. The C engines under
`examples/*/c/` are the deploy leg: they re-derive every trainer number from the
packed artifacts alone.

## The pipeline

Every example runs the same five stages in one process:

1. fp32 pretrain: Adam with cosine decay, gradient clipping, and SIMD kernels
   (`@Vector`, AVX2/NEON) where the model allows it.
2. BitNet ternary QAT: every projection is ternarized with an absmean scale
   (`scale = mean(|W|)`, trit `+1/-1` beyond `scale/2`, else `0`) and trained
   with the straight-through estimator. Gradients are masked to `|W| < scale`,
   activations are quantized per token to 8 bit, and embeddings and biases
   stay fp32.
3. Validation: fp32 against naive PTQ against QAT on a held-out split, plus
   trit histograms, latency and storage accounting.
4. Ternary packing: 5 trits per byte
   ([compilade's scheme](https://compilade.net/blog/ternary-packing), the one
   llama.cpp uses for BitNet weights). Base-3 encode with a ceiling rescale,
   then decode with a multiply and a shift instead of a division.
5. Round-trip check: reload the packed file, unpack, re-evaluate, and require
   the metric to match the pre-export model bit for bit.

### Framework modules

| module | what it provides |
| --- | --- |
| `framework/rng.zig` | deterministic xorshift stream: uniform / gauss / range |
| `framework/ternary.zig` | the 5-trits-per-byte weight codec (pack5 / unpack5) |
| `framework/quant.zig` | absmean quantization, STE grad mask, 8-bit activation quant, wired-only scale variant |
| `framework/adam.zig` | 8-wide SIMD Adam update, cosine LR schedule, bias corrections |
| `framework/packed_model.zig` | little-endian model-file writers/reader, the shared ternary-matrix record |
| `framework/timing.zig` | portable monotonic clock |

A new example needs only model-specific code: the forward and backward pass,
and an export layout over these primitives.

## Build and run

```bash
zig build --release=fast                # framework examples + fp32 baseline + C reference
zig build test                          # framework + example unit tests
zig build --release=fast example-gpt    # example 1: micro-GPT on names
zig build --release=fast example-cnn    # example 2: TinyML gesture 1D-CNN
zig build --release=fast example-ncps   # example 3: CfC/LTC Neural Circuit Policies
zig build --release=fast example-timestral  # example 4: quantize Timestral-5M
zig build --release=fast run            # fp32 baseline only (microGPT)
make deploy-check                       # C deployment check on every packed artifact
make timestral-check                    # example 4: end-to-end parity against PyTorch
make sketch                             # example 4: the drawing app (http://127.0.0.1:8000)
```

Requirements: Zig 0.16.0, `make`, and Python 3.11+ for the tools
(`examples/*/python/`, `examples/timestral_sketch/deploy/`). The Zig half has no
dependencies at all. Python extras are locked with `uv`:

```bash
uv sync --extra eval                 # figures and metrics
uv sync --extra qat --extra eval     # Timestral fine-tuning
uv sync --extra serve                # the sketch app
```

## Documentation

| document | what it answers |
| --- | --- |
| [`docs/ARTIFACT-FORMAT.md`](docs/ARTIFACT-FORMAT.md) | the five binary formats, field by field, plus the ternary codec's golden vectors |
| [`docs/TIMESTRAL.md`](docs/TIMESTRAL.md) | what 1.6 bits/weight costs Timestral-5M, measured |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | throughput, with the machine and the method |
| [`THIRD_PARTY.md`](THIRD_PARTY.md) | every upstream this is derived from, and its license |
| [`CHANGELOG.md`](CHANGELOG.md) | what changed, and which numbers moved |

## Design decisions

The five choices below shape the rest of the repository. Each paragraph states
a conclusion; the numbers behind it are in the documents above.

Zig runs the pipeline, and C appears only where the job is a second
implementation. Training and deployment in one language remove the boundary
where "it matched on my machine" stops being true. `zig build` is the build
system, the test runner and the cross-compiler, and `@Vector` emits AVX2 or
NEON from one source. The C engines stay because their job is to disagree
with Zig.

The weight codec packs five trits into one byte. A trit carries
log2(3) ≈ 1.585 bits and `3^5 = 243 ≤ 256`, so a byte holds five of them at
1.6 bits per weight, decoded with one multiply, one shift and one mask. There
is no division, which the targeted Cortex-M cores do not have. Two bits per
weight would make the file 25 % larger and would not make decoding cheaper.

Every shipped artifact gets a C engine of its own. A round-trip check shares
its reader with its writer, so a shared misreading of the format round-trips
perfectly. Each engine reads only the packed file, recomputes the metric, and
compares it with the value the trainer wrote into the artifact's footer. A
mismatch fails `make deploy-check`.

The quantizer uses absmean scales rather than a learned threshold.
`scale = mean(|W|)` with a fixed 0.5 threshold makes the artifact a
deterministic function of the weights: one command, no calibration corpus, no
optimizer state to version. Where absmean is not enough, the fix here is more
training (QAT), not a smarter quantizer.

The fp32 backend stays. A quantized number means little without its baseline
measured in the same process. Both backends answer one endpoint, so the cost
of 1.77 bits is drawn on the same chart instead of quoted in a table.

## Examples

| example | model | parameters |
| --- | --- | --- |
| [`gpt_names`](examples/gpt_names/README.md) | char-level micro-GPT on names | 4,192 |
| [`gesture_cnn`](examples/gesture_cnn/README.md) | TinyML gesture 1D-CNN | 2,524 |
| [`liquid_sine`](examples/liquid_sine/README.md) | CfC/LTC Neural Circuit Policies | 352 to 14,689 |
| [`timestral_sketch`](examples/timestral_sketch/README.md) | Timestral-5M, quantized and served | 5,208,960 |

## The deploy leg: C engines

`examples/*/c/` holds independent C deployment engines. They load the packed
artifacts and the exported val split, decode the trits, run the models
add/skip/subtract style with no per-weight float table, then re-derive the
trainer's val metric and compare:

```bash
make cnn-c         # cnn1d-c: accuracy + confusion matrix parity for the CNN
make ncps-check    # ncps-c:  val-MSE parity for all four CfC/LTC artifacts
make deploy-check  # both
```

Each engine must report the trainer's numbers exactly; a mismatch fails the
build target.

## The fp32 baseline

`examples/gpt_names/fp32.zig`, with its C twin
`examples/gpt_names/c/microgpt.c`, is the reference point the framework's fp32
pretrain leg is validated against: the original microGPT, fp32 only, trained
on the same names data. Training and inference use separate forward passes.
`gpt_forward` stores activations for backprop; `gpt_forward_infer` is a
specialised single-token path whose logits match it to within fp32 rounding.
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) covers that path and its limits.

| machine | backend | tok/sec | conditions |
| --- | --- | --- | --- |
| AMD Ryzen 9 7940H | Zig fp32, AVX2 | 8,441,952 | median of 5, 高性能 power plan, 2026-10-30 |
| AMD Ryzen 9 7940H | C fp32, AVX2 | 8,217,680 | same session, same flags |

The same two binaries measure ~4.0 M tok/sec on this laptop's 平衡 (balanced)
power plan, so the plan alone is worth about 2x, more than any code change in
this repository. Numbers without that condition are not comparable, which is
why the table carries it.

## Legacy PyTorch path

`examples/gpt_names/python/train_qat.py` is the original PyTorch QAT trainer,
and `examples/gpt_names/python/export.py` is its C-header exporter
(`examples/gpt_names/c/model.h`, consumed by
`examples/gpt_names/c/inference.c`). The Zig framework supersedes both. The
scripts stay because their checkpoint artifacts
(`examples/gpt_names/modeldata/bitnet_gpt.pth`,
`examples/gpt_names/modeldata/bitnet_gpt_weights.npz`) document the
quantization semantics the framework mirrors. The two projects the scheme grew
out of, [BitNetMCU](https://github.com/cpldcpu/BitNetMCU) and
[microGPT-C](https://github.com/vixhal-baraiya/microgpt-c), are credited in
[THIRD_PARTY.md](THIRD_PARTY.md), which also explains why neither is vendored.

## Layout

```
src/framework.zig    framework entry: re-exports the six core modules
  src/framework/       rng / ternary / quant / adam / packed_model / timing
examples/            one directory per model; each owns its code, data and artifacts
  gpt_names/           gpt_names.zig, fp32.zig (the baseline), data/, c/, python/, modeldata/, web/
  gesture_cnn/         gesture_cnn.zig, c/, modeldata/, web/
  liquid_sine/         liquid_sine.zig, c/, modeldata/
  timestral_sketch/    timestral_sketch.zig, timestral/ (the port), deploy/, python/,
                       reference/, modeldata/, web/
docs/                format spec, performance notes, the Timestral write-up
```

Every example writes its artifacts to its own `modeldata/`, and the only thing
they share is `src/framework/`.
