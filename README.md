# microGPT-C

The most atomic way to train and inference a GPT in pure, dependency-free C.

A character-level transformer with forward pass, backprop, Adam and
sampling, in one C file with nothing beyond libc. It trains on ~32k names
in a couple of seconds and generates new ones.

## Build and run

```bash
make run
```

Or run it directly, on any corpus with one item per line:

```bash
./microgpt data/names.txt
```

Builds on macOS, Linux and Windows (MSYS2), on ARM64 with NEON and x86-64
with AVX2. The Makefile picks the flags for the host.

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
