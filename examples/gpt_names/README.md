# gpt_names: a char-level micro-GPT

The tiny char-level transformer from microGPT-C (4,192 parameters: 1 layer, 4 heads,
16-dim embeddings, 64-dim MLP, ReLU², RMSNorm) is trained on about 32k names and
generates new ones. Trained and deployed ternary, the model file is about 22x smaller
than the fp32 weights.

Measured on `examples/gpt_names/data/names.txt` (80/20 train/hold-out split, 20000 fp32
steps then 5000 QAT steps, about 1 s total on a Ryzen 9 7940H):

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

Run from the repository root:

```bash
zig build --release=fast example-gpt    # writes examples/gpt_names/modeldata/tritnetmcu_packed.bin
```

The directory also holds the fp32 baseline (`fp32.zig`, a port of `c/microgpt.c`), the
legacy PyTorch QAT path (`python/`), the C engine that scores the packed file (`c/`),
and the page that compares the C and Zig implementations (`web/index.html`).
