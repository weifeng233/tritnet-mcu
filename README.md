# TritNet-MCU (tritnetmcu)

A ternary-weight micro-GPT for MCUs, trained end to end in pure Zig with
nothing beyond libc: fp32 pretrain, BitNet-style quantization-aware
training, held-out validation, and a ternary-packed model export — one
binary, one command, about a second on a laptop.

The network is the tiny char-level transformer from microGPT-C (4,192
parameters: 1 layer, 4 heads, 16-dim embeddings, 64-dim MLP, ReLU²,
RMSNorm) that trains on ~32k names and generates new ones. Every linear
projection is then quantized to ternary weights {-1, 0, +1} and stored at
1.6 bits per weight.

## Layout

```
src/            Zig core — the architecture
  tritnet.zig     full pipeline: fp32 pretrain -> ternary QAT -> packed export
  fp32.zig        fp32-only trainer/inference baseline (the original microGPT)
  ternary.zig     ternary packing (5 trits/byte) + exhaustive unit tests
c/              C reference implementations
  microgpt.c      original fp32 C GPT, kept for A/B comparison
  inference.c/h   C inference engine for the QAT checkpoint
  model.h         generated ternary model header
python/         legacy PyTorch QAT path (train_qat.py + export.py)
reference/      vendored references: BitNetMCU/, mlp-mcu/
web/            single-page visualization of the C-vs-Zig comparison
data/ modeldata/ docs/
```

## Build and run

```bash
zig build --release=fast            # installs tritnetmcu, tritnet-fp32, microgpt-c
zig build test                      # ternary packing unit tests
zig build --release=fast tritnet    # full pipeline (or: ./zig-out/bin/tritnetmcu data/names.txt)
zig build --release=fast run        # fp32 baseline only
make CC=gcc run                     # C reference with a plain toolchain
```

## The pipeline

`tritnetmcu` runs, in one process:

1. **fp32 pretrain** — identical math to the fp32 baseline: Adam, cosine
   schedule, gradient clipping, SIMD (AVX2/NEON via `@Vector`) kernels.
2. **BitNet ternary QAT** — every projection is ternarized with an absmean
   scale (`scale = mean(|W|)`, trit `+1/-1` beyond `scale/2`, else `0`) and
   trained with the straight-through estimator; gradients are masked to
   `|W| < scale` (mirroring the legacy PyTorch path), with per-token 8-bit
   activation quantization. Embeddings stay fp32.
3. **Validation** — held-out loss for fp32 vs naive PTQ vs QAT, plus
   generation samples from each.
4. **Ternary packing** — 5 trits per byte
   ([compilade's scheme](https://compilade.net/blog/ternary-packing), the
   one llama.cpp uses for BitNet models): base-3 with a ceiling rescale on
   encode, division-free multiply-and-shift on decode.
   `modeldata/tritnetmcu_packed.bin` is written, reloaded, and re-evaluated
   to prove the packed artifact is lossless.

Measured on data/names.txt (80/20 train/hold-out split, 20000 fp32 steps
then 5000 QAT steps, ~1 s total on a Ryzen 5 5600H):

```
eval (avg nats/char, float activations)
  fp32  train 2.1947   val 2.1990
  ptq   val 2.4967   (ternary, no QAT)
  qat   train 2.2728   val 2.2708
summary
  fp32 val 2.1990 | ptq val 2.4967 | qat val 2.2708 | qat recovery 75.9% of fp32->ptq gap

storage
  ternary weights      :   3504 params
  fp32 weights         :  16768 bytes
  naive 2-bit          :    876 bytes
  packed 5-trits/byte  :    761 bytes  (22.0x smaller than fp32, 1.74 bits/trit incl. scales)
  model file           :   3579 bytes (embeddings kept fp32)
packed round-trip
  val loss pre-export  : 2.270826
  val loss reloaded    : 2.270826
  round-trip: OK (lossless)
```

Naive post-training quantization costs +0.30 nats/char; QAT claws back
three quarters of that while making every projection ternary, and the
packed file ships the result at ~22x below fp32 weight storage.

## The fp32 baseline

The fp32 path (and its C twin) generalise rather than memorise: trained on
20000 of the 32033 names, the fp32 model scores 2.2054 nats per character
on those and 2.2039 on the 12033 it never saw, beating an interpolated
trigram with nearly five times as many parameters. Training and inference
use separate forward passes — `gpt_forward` stores activations for
backprop, `gpt_forward_infer` is a specialised single-token path whose
logits match it to within fp32 rounding.
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) covers how that path works and
what limits it.

| machine | backend | tok/sec |
| --- | --- | --- |
| Apple M5 Pro | NEON | 10,168,430 |
| AMD Ryzen 5 5600H | AVX2 | 6,927,775 |

## Legacy PyTorch path

`python/train_qat.py` is the original PyTorch QAT trainer and
`python/export.py` its C-header exporter (`c/model.h`, consumed by
`c/inference.c`). The Zig pipeline supersedes it; the scripts are kept
because their checkpoint artifacts (`modeldata/bitnet_gpt.pth`,
`modeldata/bitnet_gpt_weights.npz`) document the quantization semantics
the Zig implementation mirrors. `reference/` holds the vendored
BitNetMCU and mlp-mcu projects that inspired the scheme.
