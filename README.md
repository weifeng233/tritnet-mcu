# microGPT-C

The most atomic way to train and inference a GPT in pure, dependency-free C.

A character-level transformer with forward pass, backprop, Adam and
sampling, in one C file with nothing beyond libc. It trains on ~32k names
in a couple of seconds and generates new ones.

## Build and run

```bash
zig build --release=fast run
```

Or with an explicit corpus:

```bash
./zig-out/bin/microgpt data/names.txt
```

The project is now pure Zig (`src/microgpt.zig`, `build.zig`). The C source
(`src/microgpt.c`) is kept as a reference and is still built as `microgpt-c`
by `zig build --release=fast` for A/B comparison. The Makefile remains for
`make run` / `make CC=gcc` compatibility and also defaults to `zig cc`.

Builds on macOS, Linux and Windows (MSYS2), on ARM64 with NEON and x86-64
with AVX2. `zig build --release=fast` maps to `-Ofast` (`ReleaseFast`) and
`zig cc` remains available via `make`.

```
step 5000 / 20000 | loss 2.6036  (avg 2.2940)
step 10000 / 20000 | loss 1.9639  (avg 2.2564)
step 15000 / 20000 | loss 2.7007  (avg 2.2151)
step 20000 / 20000 | loss 2.3463  (avg 2.2201)

inference
sample  1: kayley
sample  2: maria
sample  3: arana
sample  4: shayan
sample  5: jayden
sample  6: saria
sample  7: kaylen
sample  8: amari
sample  9: alina
sample 10: mailyn
  c fp32+NEON       10168430 tok/sec
```

## Notes

The model has 4192 parameters and generalises rather than memorises.
Trained on 20000 of the 32033 names, it scores 2.2054 nats per character
on those and 2.2039 on the 12033 it never saw, beating an interpolated
trigram that has nearly five times as many parameters.

Training and inference use separate forward passes. `gpt_forward` stores
activations for backprop; `gpt_forward_infer` is a specialised
single-token path whose logits match it to within fp32 rounding.
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) covers how that path works and
what limits it.

| machine | backend | tok/sec |
| --- | --- | --- |
| Apple M5 Pro | NEON | 10,168,430 |
| AMD Ryzen 5 5600H | AVX2 | 6,927,775 |

## BitNet ternary QAT + packing, full pipeline in Zig

`src/bitnet.zig` (built as `bitnet`) runs the entire lifecycle in one
dependency-free binary: fp32 pretrain, BitNet quantization-aware training,
held-out validation, and a ternary-packed model export.

```bash
zig build --release=fast bitnet            # or: ./zig-out/bin/bitnet data/names.txt [steps] [qat_steps]
zig build test                             # ternary packing unit tests
```

Quantization follows BitNet b1.58: each weight matrix is ternarized with an
absmean scale (`scale = mean(|W|)`, trit `+1/-1` beyond `scale/2`, else `0`)
and trained with the straight-through estimator, the gradient masked to
`|W| < scale` (mirroring `bitnet_gpt_train.py`), plus per-token 8-bit
activation quantization during QAT. Embeddings stay fp32.

Validation on data/names.txt (80/20 train/hold-out split, 20000 fp32 steps
then 5000 QAT steps, ~1 s total on a Ryzen 5 5600H):

```
eval (avg nats/char, float activations)
  fp32  train 2.1947   val 2.1990
  ptq   val 2.4967   (ternary, no QAT)
  qat   train 2.2728   val 2.2708
summary
  fp32 val 2.1990 | ptq val 2.4967 | qat val 2.2708 | qat recovery 75.9% of fp32->ptq gap
```

Naive post-training quantization costs +0.30 nats/char; QAT claws back
three quarters of that while making every projection ternary.

Storage uses [compilade's ternary packing](https://compilade.net/blog/ternary-packing)
(the scheme llama.cpp uses for BitNet models): five trits per byte via base-3
with a ceiling rescale, decoded division-free by peeling digits off the top
with a multiply-and-shift. `src/ternary.zig` implements pack/unpack with
exhaustive round-trip tests over all 243 blocks; the pipeline exports
`modeldata/bitnet_gpt_packed.bin`, reloads it, and re-evaluates to prove the
packed artifact is bit-exact:

```
storage
  ternary weights      :   3504 params
  fp32 weights         :  16768 bytes
  naive 2-bit          :    876 bytes
  packed 5-trits/byte  :    761 bytes  (22.0x smaller than fp32, 1.74 bits/trit incl. scales)
  model file ...       :   3579 bytes (embeddings kept fp32)
packed round-trip
  val loss pre-export  : 2.270826
  val loss reloaded    : 2.270826
  round-trip: OK (lossless)
```
