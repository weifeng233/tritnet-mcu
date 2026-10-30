# TritNet-MCU (tritnetmcu)

English | [简体中文](README.zh-CN.md)

A pure-Zig **train → quantize → deploy framework** for ternary-weight
{-1, 0, +1} MCU models, plus three worked examples. Nothing beyond libc:
one command runs fp32 pretraining, BitNet-style quantization-aware
training, held-out validation, a ternary-packed model export, and a
round-trip check that the packed artifact is lossless — seconds per
example, on a laptop.

The framework lives in `src/framework.zig` (RNG, ternary codec, absmean
QAT quantizer, Adam + cosine schedule, packed-model container, timing).
The `examples/` directory drives it with three deliberately different
model families; the C engines in `c/` are the deploy leg, re-deriving
every trainer number from the packed artifacts alone.

## The pipeline

Every example runs the same five stages in one process:

1. **fp32 pretrain** — Adam with cosine decay, gradient clipping, SIMD
   (`@Vector`, AVX2/NEON) kernels where the model allows it.
2. **BitNet ternary QAT** — every projection is ternarized with an
   absmean scale (`scale = mean(|W|)`, trit `+1/-1` beyond `scale/2`,
   else `0`) and trained with the straight-through estimator; gradients
   are masked to `|W| < scale`, activations are quantized per token to
   8 bit, embeddings/biases stay fp32.
3. **Validation** — fp32 vs naive PTQ vs QAT on a held-out split, plus
   trit histograms, latency and storage accounting.
4. **Ternary packing** — 5 trits per byte
   ([compilade's scheme](https://compilade.net/blog/ternary-packing), the
   one llama.cpp uses for BitNet weights): base-3 encode with a ceiling
   rescale, division-free multiply-and-shift decode.
5. **Round-trip check** — reload the packed file, unpack, re-evaluate;
   the metric must match the pre-export model bit-for-bit.

### Framework modules

| module | what it provides |
| --- | --- |
| `framework/rng.zig` | deterministic xorshift stream: uniform / gauss / range |
| `framework/ternary.zig` | the 5-trits-per-byte weight codec (pack5 / unpack5) |
| `framework/quant.zig` | absmean quantization, STE grad mask, 8-bit activation quant, wired-only scale variant |
| `framework/adam.zig` | 8-wide SIMD Adam update, cosine LR schedule, bias corrections |
| `framework/packed_model.zig` | little-endian model-file writers/reader, the shared ternary-matrix record |
| `framework/timing.zig` | portable monotonic clock |

A new example needs only model-specific code: the forward/backward pass
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

## Example 1 — `gpt_names`: char-level micro-GPT

The flagship walkthrough: the tiny char-level transformer from
microGPT-C (4,192 parameters: 1 layer, 4 heads, 16-dim embeddings,
64-dim MLP, ReLU², RMSNorm) trained on ~32k names, generating new ones.
Trained and deployed ternary, the model file is ~22x smaller than the
fp32 weights.

Measured on data/names.txt (80/20 train/hold-out split, 20000 fp32
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
zig build --release=fast example-gpt    # writes modeldata/tritnetmcu_packed.bin
```

## Example 2 — `gesture_cnn`: TinyML sensor classifier

The canonical TinyML 1D-CNN task (EdgeImpulse / MLPerf-Tiny style):
4-class gesture recognition — rest / shake / circle / tap — from 3-axis
accelerometer windows, generated deterministically from a fixed seed
(random phase, amplitude jitter, per-sample noise), so the run is
reproducible with no external dataset. 2,524 parameters, 54,208 MACs
per inference:

```
input 3x128
Conv1D(3 -> 8, k=8)  + ReLU + MaxPool(2)   -> 8x60
Conv1D(8 -> 16, k=4) + ReLU + MaxPool(2)   -> 16x28
Flatten -> Dense(448 -> 4) + softmax
```

Biases stay fp32 (28 of 2,524 parameters); every conv kernel and the
dense weight go ternary. Unit tests in the example cover the conv/pool
kernels against direct references and an end-to-end numerical gradient
check.

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
  ternary weights      :  41.1 us/inference  (1320M MACs/s)

packed round-trip: OK (lossless)   qat recovery 99.9% of fp32->ptq loss gap
```

Naive ternarization collapses the val accuracy to 65.75%; QAT restores
all 100% while every kernel runs as {-1, 0, +1} trits at 1.6 bits per
weight.

```bash
zig build --release=fast example-cnn    # writes modeldata/cnn1d_packed.bin + cnn1d_valset.bin
make cnn-c                              # C deployment check (or: make deploy-check)
```

## Example 3 — `liquid_sine`: Neural Circuit Policies (CfC/LTC)

A port of the [ncps](https://github.com/mlech26l/ncps) stack (Lechner &
Hasani) onto the framework: continuous-time RNNs whose sparse "wiring"
is a circuit of sensory, inter, command and motor neurons. Implemented
from the reference:

- `wirings.NCP` / `AutoNCP` — the stochastic 3-layer circuit constructor
  (sensory fanout, inter fanout, recurrent command synapses, motor fanin,
  ±1 polarities, reachability repair), plus `wirings.FullyConnected`
- `torch.CfCCell` in all three modes (default / pure / no_gate) with
  per-step timespans, and the `WiredCfCCell` layer stack whose ff1/ff2
  projections are masked by the wiring adjacency — the time-gate
  projections stay dense, exactly as in ncps
- `torch.LTCCell` — the original liquid time-constant ODE solved with 6
  Euler fixed-point unfolds per step; BPTT runs the chain rule back
  through every unfold (finite-difference checked)
- the `torch_cfc_sinusoidal.py` task (sin/cos in, double-frequency sine
  out, MSE) generalized into a 1,600-sequence dataset with random phase
  and jittered frequency

Task: 48-step sequences, input [sin, cos] over [0, 3·pi·k], target the
same waveform at twice the frequency. Two details make the wiring
survive quantization: the sparsity is baked into the trits themselves
(masked synapses have zero latents, hence zero trits — the packed
artifact needs no side-car mask), and the absmean scale is computed over
*wired* synapses only, so the zeros of the sparse circuit never dilute
the scale. Four models train in one run (mirroring the ncps sinusoidal
example):

| model | params | fp32 val MSE | ptq val | qat val | qat recovery | packed |
| --- | --- | --- | --- | --- | --- | --- |
| CfC(2→32, backbone 64, proj 1) | 14,689 | 0.000520 | 0.448876 | 0.007895 | 98.4% | 3,779 B (15.5x < fp32) |
| WiredCfC + FullyConnected(8, out 1) | 352 | 0.002416 | 0.354496 | 0.029022 | 92.4% | 280 B (5.0x) |
| WiredCfC + NCP(16 inter, 8 cmd, 1 motor) | 2,056 | 0.001180 | 0.514103 | 0.019580 | 96.4% | 964 B (8.5x) |
| LTC + NCP(16 inter, 8 cmd, 1 motor) | 2,781 | 0.000815 | 0.544995 | 0.001454 | 99.9% | 8,623 B |

Measured on a Ryzen 9 7940H (1,600 sequences, 80/20 split, ~4.5 min
total). The liquid cell is the star: naive PTQ costs it 0.545 MSE — its
continuous-time dynamics are fragile to ternarization — yet QAT claws
back 99.9% of the gap to 0.001454, beating the same-size wired CfC by
~13x while running a 48-step inference in 2.8 ms on the ODE kernels.
(The LTC packs to only 1.3x smaller than fp32 because most of its
parameters are the fp32 physical ODE parameters — only the 675 synapse
strengths are ternary.) Every packed artifact round-trips losslessly,
and on the exact ncps example sequence the QAT CfC tracks the
double-frequency sine at 0.000603 MSE under deployment weights.

Deliberate deviations from the reference, documented in
`examples/liquid_sine.zig`: the wiring RNG is the framework's xorshift
seeded per wiring (same construction algorithm, different concrete
synapse choices), weights use Xavier-Gauss init, the unwired backbone is
64 units instead of 128, and mixed_memory (the fp32 LSTM augmentation)
is not ported.

```bash
zig build --release=fast example-ncps   # writes modeldata/ncps_{cfc,fc,ncp,ltc}_packed.bin + ncps_valset.bin
make ncps-check                         # C deployment check on all four artifacts
```

## Example 4 — `timestral_sketch`: deploying Mistral's Timestral-5M, then ternarizing it

Not every model this repository deploys was trained here. This example stands up
[Mistral AI's Timestral-5M](https://huggingface.co/mistralai/Timestral-5M) — 5.2 M
parameters, 8 patch-transformer blocks, 21 quantiles per step — behind a *sketch-a-series*
app: draw a stroke and the model continues it as a quantile fan, the interaction of
[the Space](https://huggingface.co/spaces/townbox/sketch-a-series). Then it quantizes that
model with this framework's own stack — absmean QAT, 5 trits per byte, a packed artifact,
a Zig engine — and runs the **same page** against both backends so the price of 1.77 bits
per weight is visible side by side.

The Zig engine is a full port of the forward pass (per-patch causal normalization, `[Enc ;
Dec]` attention with RoPE and sigmoid gates, SwiGLU, the monotone quantile head, the
advancing refine frame, clampsym) and reproduces PyTorch to 2.4·10⁻⁶ relative error on the
fp32 weights, which is what makes the comparison mean quantization and nothing else.

```
fp32 (safetensors, PyTorch)      20,843,960 B
ternary artifact (Zig, 1.77 bit)  1,152,070 B   (18.1x smaller)

24 held-out strokes, 256-step context, 96-step horizon
  variant      pinball   MAE     MASE    cov90   one request
  fp32          0.0711   0.217   5.89    0.961     31-44 ms
  PTQ ternary   0.2023   0.522  17.55    0.777        -
  QAT ternary   0.1071   0.293   8.91    0.901     16-22 ms
```

Naive ternarization triples the error; 2,500 steps of quantization-aware fine-tuning
(9.5 min on four CPU threads, `python/timestral_qat.py`) recover 73 % of that gap. What
survives best is what a sketch mostly is: trends and noisy walks come out within 10 % of
fp32, while strokes whose continuation needs an amplitude — drawn curves and sinusoids —
come out visibly damped (2.2–2.8× the pinball). The app draws both medians on the same
chart so that cost is visible rather than hidden.

```bash
make fetch-timestral                 # the released checkpoint (21 MB)
make timestral-qat quantize-timestral   # fine-tune, then quantize + calibrate
make timestral-check                 # parity against PyTorch on the same strokes
make sketch                          # http://127.0.0.1:8000, both backends
```

The page (`web/sketch-a-series.html`), the server (`deploy/app.py`) and the whole story —
artifact format, what breaks, why the gradient mask had to go, the quantile-width
calibration — are in [docs/TIMESTRAL.md](docs/TIMESTRAL.md) and
[deploy/README.md](deploy/README.md).

## The deploy leg: C engines

`c/` holds independent C deployment engines — they load the packed
artifacts plus the exported val split, decode the trits and run the
models add/skip/subtract style (no per-weight float table), then
re-derive the trainer's val metric and compare:

```bash
make cnn-c         # cnn1d-c: accuracy + confusion matrix parity for the CNN
make ncps-check    # ncps-c:  val-MSE parity for all four CfC/LTC artifacts
make deploy-check  # both
```

Each engine must report the trainer's numbers exactly; a mismatch fails
the build target.

## The fp32 baseline

`src/fp32.zig` (with its C twin `c/microgpt.c`) is the reference point
the framework's fp32 pretrain leg is validated against — the original
microGPT, fp32 only, trained on the same names data. Training and
inference use separate forward passes — `gpt_forward` stores activations
for backprop, `gpt_forward_infer` is a specialised single-token path
whose logits match it to within fp32 rounding.
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) covers how that path works
and what limits it ([中文版](docs/PERFORMANCE.zh-CN.md)).

| machine | backend | tok/sec |
| --- | --- | --- |
| AMD Ryzen 9 7940H | AVX2 | 5,491,565 |

## Legacy PyTorch path

`python/train_qat.py` is the original PyTorch QAT trainer and
`python/export.py` its C-header exporter (`c/model.h`, consumed by
`c/inference.c`). The Zig framework supersedes it; the scripts are kept
because their checkpoint artifacts (`modeldata/bitnet_gpt.pth`,
`modeldata/bitnet_gpt_weights.npz`) document the quantization semantics
the framework mirrors. The two projects that inspired the scheme,
[BitNetMCU](https://github.com/cpldcpu/BitNetMCU) and
[microGPT-C](https://github.com/vixhal-baraiya/microgpt-c), are credited in
[THIRD_PARTY.md](THIRD_PARTY.md); see there for why neither is vendored.

## Layout

```
src/framework.zig    framework entry: re-exports the six core modules
  src/framework/       rng / ternary / quant / adam / packed_model / timing
src/fp32.zig         fp32 reference baseline (the original microGPT)
src/timestral.zig    example 4: Timestral-5M deployment module
  src/timestral/       model / engine / weights / artifact / eval / fixture
examples/            worked examples, one binary each
  gpt_names.zig        char-level micro-GPT on names.txt
  gesture_cnn.zig      TinyML 1D-CNN gesture classifier
  liquid_sine.zig      CfC/LTC Neural Circuit Policies, sine doubling
  timestral_sketch.zig Timestral-5M: quantize / eval / forecast / serve
c/                   C deployment engines (cnn1d, ncps) + fp32 C reference
deploy/              the sketch-a-series server (FastAPI) + its README
python/              legacy PyTorch QAT path, plus the Timestral tools (QAT, parity, figures)
reference/           vendored reference: timestral/ (Apache-2.0), sketch-a-series/
web/                 single-page visualizations and the sketch-a-series app
data/ modeldata/ docs/
```
