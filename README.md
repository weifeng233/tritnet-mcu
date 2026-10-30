# TritNet-MCU (tritnetmcu)

[![ci](https://github.com/weifeng233/tritnet-mcu/actions/workflows/ci.yml/badge.svg)](https://github.com/weifeng233/tritnet-mcu/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

English | [简体中文](README.zh-CN.md)

A pure-Zig train, quantize and deploy framework for ternary-weight {-1, 0, +1}
MCU models, with four worked examples. It needs nothing beyond libc: one
command runs fp32 pretraining, BitNet-style quantization-aware training,
held-out validation, a ternary-packed model export, and a round-trip check
that the packed file is lossless. Each example takes seconds on a laptop.

The framework lives in `src/framework.zig` (RNG, ternary codec, absmean QAT
quantizer, Adam with a cosine schedule, the packed-model container, timing).
The `examples/` directory drives it with four different model families. The C
engines under `examples/*/c/` are the deploy leg: they re-derive every trainer
number from the packed artifacts alone.

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

## Example 1: `gpt_names`, a char-level micro-GPT

The tiny char-level transformer from microGPT-C (4,192 parameters: 1 layer,
4 heads, 16-dim embeddings, 64-dim MLP, ReLU², RMSNorm) is trained on ~32k
names and generates new ones. Trained and deployed ternary, the model file is
~22x smaller than the fp32 weights.

Measured on examples/gpt_names/data/names.txt (80/20 train/hold-out split, 20000 fp32
steps then 5000 QAT steps, ~1 s total on a Ryzen 9 7940H):

```
eval (avg nats/char, float activations)
  fp32  train 2.1944   val 2.1987
  ptq   val 2.4948   (ternary, no QAT)
  qat   train 2.2722   val 2.2700
  trits: -1 35.8%   0 26.5%   +1 37.7%

storage
  packed 5-trits/byte  :    761 bytes  (22.0x smaller than fp32, 1.74 bits/trit incl. scales)
  model file           :   3579 bytes (embeddings kept fp32)
packed round-trip: OK (lossless)

summary
  fp32 val 2.1987 | ptq val 2.4948 | qat val 2.2700 | qat recovery 75.9% of fp32->ptq gap
```

```bash
zig build --release=fast example-gpt    # writes examples/gpt_names/modeldata/tritnetmcu_packed.bin
```

## Example 2: `gesture_cnn`, a TinyML sensor classifier

The 1D-CNN task from EdgeImpulse and MLPerf-Tiny: four gesture classes (rest,
shake, circle, tap) from 3-axis accelerometer windows, generated
deterministically from a fixed seed with random phase, amplitude jitter and
per-sample noise, so the run needs no external dataset. The model has 2,524
parameters and 54,208 MACs per inference:

```
input 3x128
Conv1D(3 -> 8, k=8)  + ReLU + MaxPool(2)   -> 8x60
Conv1D(8 -> 16, k=4) + ReLU + MaxPool(2)   -> 16x28
Flatten -> Dense(448 -> 4) + softmax
```

Biases stay fp32 (28 of 2,524 parameters); every conv kernel and the dense
weight go ternary. Each conv and pool kernel is tested against a direct
reference, and the example also runs a numerical gradient check end to end.

Measured on a Ryzen 9 7940H (4000 samples, 80/20 split, 2000 fp32
steps then 1500 QAT steps, ~11 s total):

```
eval (accuracy, float activations)
  fp32  train 100.00%   val 100.00%   (val loss 0.0014)
  ptq   val 65.75%   (ternary, no QAT)
  qat   train 100.00%   val 100.00%
  trits: -1 42.6%   0 21.2%   +1 36.2%

storage
  packed 5-trits/byte  :    672 bytes  (15.0x smaller than fp32)

latency (host CPU, float activations)
  fp32 weights         :  45.67 us/inference  (1187.0M MACs/s)
  ternary weights      :  42.93 us/inference  (1262.6M MACs/s)

packed round-trip: OK (lossless)   qat recovery 99.9% of fp32->ptq loss gap
```

Naive ternarization drops val accuracy to 65.75 %; QAT brings it back to
100 % with every kernel running as {-1, 0, +1} trits at 1.6 bits per weight.

```bash
zig build --release=fast example-cnn    # writes examples/gesture_cnn/modeldata/cnn1d_packed.bin + cnn1d_valset.bin
make cnn-c                              # C deployment check (or: make deploy-check)
```

## Example 3: `liquid_sine`, Neural Circuit Policies (CfC/LTC)

A port of the [ncps](https://github.com/mlech26l/ncps) stack (Lechner &
Hasani) onto the framework: continuous-time RNNs whose sparse wiring forms a
circuit of sensory, inter, command and motor neurons. Implemented from the
reference:

- `wirings.NCP` / `AutoNCP`: the stochastic 3-layer circuit constructor
  (sensory fanout, inter fanout, recurrent command synapses, motor fanin,
  ±1 polarities, reachability repair), plus `wirings.FullyConnected`
- `torch.CfCCell` in all three modes (default / pure / no_gate) with per-step
  timespans, and the `WiredCfCCell` layer stack whose ff1/ff2 projections are
  masked by the wiring adjacency; the time-gate projections stay dense, as in
  ncps
- `torch.LTCCell`: the original liquid time-constant ODE, solved with 6 Euler
  fixed-point unfolds per step. BPTT runs the chain rule back through every
  unfold, and a finite-difference check covers it.
- the `torch_cfc_sinusoidal.py` task (sin/cos in, double-frequency sine out,
  MSE), generalized into a 1,600-sequence dataset with random phase and
  jittered frequency

Each sequence is 48 steps of [sin, cos] over [0, 3·pi·k], and the target is
the same waveform at twice the frequency. Two details keep the wiring intact
through quantization: the sparsity is baked into the trits themselves (masked
synapses have zero latents and therefore zero trits, so the packed artifact
needs no side-car mask), and the absmean scale covers *wired* synapses only,
so the zeros of the sparse circuit never dilute it. Four models train in one
run, as in the ncps sinusoidal example:

| model | params | fp32 val MSE | ptq val | qat val | qat recovery | packed |
| --- | --- | --- | --- | --- | --- | --- |
| CfC(2→32, backbone 64, proj 1) | 14,689 | 0.000520 | 0.448876 | 0.007895 | 98.4% | 3,779 B (15.5x < fp32) |
| WiredCfC + FullyConnected(8, out 1) | 352 | 0.002416 | 0.354496 | 0.029022 | 92.4% | 280 B (5.0x) |
| WiredCfC + NCP(16 inter, 8 cmd, 1 motor) | 2,056 | 0.001180 | 0.514103 | 0.019580 | 96.4% | 964 B (8.5x) |
| LTC + NCP(16 inter, 8 cmd, 1 motor) | 2,781 | 0.000815 | 0.544995 | 0.001454 | 99.9% | 8,623 B |

Measured on a Ryzen 9 7940H (1,600 sequences, 80/20 split, ~4.5 min total).
The LTC cell handles ternarization best: naive PTQ costs it 0.545 MSE, because
its continuous-time dynamics are fragile, and QAT recovers 99.9 % of that gap
down to 0.001454. That beats the same-size wired CfC by ~13x, while a 48-step
inference runs in 2.8 ms on the ODE kernels. (The LTC packs to only 1.3x
smaller than fp32 because most of its parameters are the fp32 physical ODE
parameters; only the 675 synapse strengths are ternary.) Every packed artifact
round-trips losslessly, and on the exact ncps example sequence the QAT CfC
tracks the double-frequency sine at 0.000603 MSE under deployment weights.

Deliberate deviations from the reference are documented in
`examples/liquid_sine/liquid_sine.zig`: the wiring RNG is the framework's
xorshift seeded per wiring (same construction algorithm, different concrete
synapse choices), weights use Xavier-Gauss init, the unwired backbone has 64
units instead of 128, and mixed_memory, the fp32 LSTM augmentation, is not
ported.

```bash
zig build --release=fast example-ncps   # writes examples/liquid_sine/modeldata/ncps_{cfc,fc,ncp,ltc}_packed.bin + ncps_valset.bin
make ncps-check                         # C deployment check on all four artifacts
```

## Example 4: `timestral_sketch`, deploying Mistral's Timestral-5M and then ternarizing it

This example deploys a model trained elsewhere. It stands up
[Mistral AI's Timestral-5M](https://huggingface.co/mistralai/Timestral-5M)
(5.2 M parameters, 8 patch-transformer blocks, 21 quantiles per step) behind a
*sketch-a-series* app: draw a stroke and the model continues it as a quantile
fan, the interaction of
[the Space](https://huggingface.co/spaces/townbox/sketch-a-series). Then it
quantizes that model with this framework's own stack (absmean QAT, 5 trits per
byte, a packed artifact, a Zig engine) and runs the same page against both
backends, so the cost of 1.77 bits per weight is visible side by side.

The Zig engine is a full port of the forward pass: per-patch causal
normalization, `[Enc ; Dec]` attention with RoPE and sigmoid gates, SwiGLU,
the monotone quantile head, the advancing refine frame, and clampsym. It
reproduces PyTorch to 2.4·10⁻⁶ relative error on the fp32 weights, so the gap
that follows is quantization and not a porting bug.

```
fp32 (safetensors, PyTorch)      20,843,960 B
ternary artifact (Zig, 1.77 bit)  1,152,070 B   (18.1x smaller)

24 held-out strokes, 256-step context, 96-step horizon
  variant      pinball   MAE     MASE    cov90   one request
  fp32          0.0711   0.217   5.89    0.961     31-44 ms
  PTQ ternary   0.2023   0.522  17.55    0.777        -
  QAT ternary   0.1071   0.293   8.91    0.901     16-22 ms
```

Naive ternarization triples the error. 2,500 steps of quantization-aware
fine-tuning (9.5 min on four CPU threads,
`examples/timestral_sketch/python/timestral_qat.py`) recover 73 % of that gap.
What survives best is what a sketch mostly is: trends and noisy walks come out
within 10 % of fp32, while strokes whose continuation needs an amplitude, such
as drawn curves and sinusoids, come out visibly damped at 2.2-2.8× the
pinball. The app draws both medians on the same chart so the cost is visible.

```bash
make fetch-timestral                 # the released checkpoint (21 MB)
make timestral-qat quantize-timestral   # fine-tune, then quantize + calibrate
make timestral-check                 # parity against PyTorch on the same strokes
make sketch                          # http://127.0.0.1:8000, both backends
```

The page (`examples/timestral_sketch/web/sketch-a-series.html`), the server
(`examples/timestral_sketch/deploy/app.py`) and the whole story (artifact
format, what breaks, why the gradient mask had to go, the quantile-width
calibration) are in [docs/TIMESTRAL.md](docs/TIMESTRAL.md) and
[examples/timestral_sketch/deploy/README.md](examples/timestral_sketch/deploy/README.md).

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
