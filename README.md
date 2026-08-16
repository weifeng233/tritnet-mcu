# microGPT-C

The most atomic way to train and inference a GPT in pure, dependency-free C.

A character-level transformer with forward pass, backprop, Adam and
sampling, in one C file with nothing beyond libc. It trains on ~32k names
and generates new ones.

## Layout

```
microgpt-c/
├── Makefile
├── LICENSE
├── README.md
├── data/
│   └── names.txt        training corpus, one name per line
└── src/
    └── microgpt.c       model, training, inference
```

One source file is intentional. The whole thing should be readable in a
single sitting.

## Build and run

```bash
make run
```

The Makefile picks flags for the host: `-march=native` on AArch64, and
`-march=native -mavx2 -mfma` on x86-64. The extra x86 flags are needed
because clang ignores the in-file `#pragma GCC target`.

Run it directly, or point it at a different corpus:

```bash
./microgpt data/names.txt
```

The path defaults to `data/names.txt`. Any file with one item per line
works.

Flags:

- `-O3` maximum optimisation
- `-march=native` host-specific instructions (AVX2 or NEON)
- `-ffast-math` faster floating point, trades some precision
- `-lm` link the math library

Cross-compiling means overriding two variables. An x86-64 binary from an
Apple Silicon host:

```bash
make CC="clang -arch x86_64" ARCHFLAGS="-mavx2 -mfma"
```

Expect the loss to fall, then samples:

```
step  5000 / 20000 | loss 2.6036  (avg 2.2940)
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
  c fp32+NEON       10128264 tok/sec
```

## Is it learning?

Training uses `docs[step % num_docs]` for 20000 steps against 32033
names, so everything past index 20000 is never seen. Evaluated on that
held-out split, with the same loss convention (nats per character):

| model | train | held-out | params |
| --- | --- | --- | --- |
| uniform | 3.2958 | 3.2958 | 0 |
| unigram | 2.7636 | 2.9237 | 27 |
| bigram | 2.3854 | 2.6514 | 729 |
| interpolated trigram | 2.1548 | 2.4945 | 19683 |
| microGPT-C | 2.2054 | 2.2039 | 4192 |

The trigram beats the model on training data and loses badly on held-out
data, which is what memorising looks like. The model's two numbers are
the same to within noise, so it generalises. Held-out perplexity is 9.06
against a 27-symbol vocabulary.

Sampling 4000 names at temperature 1.0 gives 93% distinct and 74% that
are not in the dataset at all. Note that the demo prints at temperature
0.5, which sharpens the distribution and makes the output look far more
repetitive than the model actually is.

## Inference path

Training and inference use separate forward passes. `gpt_forward` stores
~1.3 KB of activations per token for backprop. `gpt_forward_infer` needs
none of that and is built for one-token-at-a-time decoding. Its logits
match the training forward to 1.4e-06 absolute on NEON and 2.4e-06 on
AVX2, against logits of magnitude ~8, which is fp32 rounding noise.

There are two backends. Weight packing, the `(token, pos)` table and the
sampler are shared; only the kernels and the forward body differ. NEON
works 4 floats wide and AVX2 8, so AVX2 needs half the instructions per
matvec: one column of a 16-row block costs a broadcast and two FMAs,
against four lane-indexed FMAs on NEON.

What the inference path does differently:

- Column-major weights, so matvecs accumulate straight into output
  registers with no horizontal reduction anywhere.
  `infer_pack_weights()` builds the transposed copies once, after the
  last optimiser step.
- Split accumulators. One accumulator set turns a matvec into a serial
  chain of `nin` FMAs. Each kernel uses 2 to 4 sets.
- Hoisted prefix. The embedding, both RMS norms and layer 0 Q/K/V depend
  only on `(token, pos)`, and there are only `vocab * BLOCK_SIZE`
  distinct inputs, so they go in a table.
- Deferred RMS scale. The MLP norm scale is a positive scalar and
  fc1/ReLU^2/fc2 are positively homogeneous, so it factors out to one
  `s^2` at the end, off fc1's critical path.
- Fused sampling. Softmax, temperature and weighted choice in one pass.
  The old path normalised probabilities only for `weighted_choice` to
  re-sum them.
- Vocab padded to 4 rather than 16. lm_head emits 16/8/4-row tiles, so a
  27-token vocab costs 28 rows instead of 32.

## What limits it

About half the time is the MLP, roughly 240 of 500 cycles per token.
Four explanations were tested and none of them held up:

| hypothesis | test | result |
| --- | --- | --- |
| load-bound (loads:FMA are 1:1) | halve weight bytes via FMLAL | 241 to 242 cyc |
| fc1/ReLU^2/fc2 chain | delete the dependency outright | -37 cyc (7%) |
| too few accumulator chains | 8 to 16 chains in fc1 | +0.7% |
| token-to-token serialisation | replay a recorded trajectory | slower |

The real limit is issue width. Extra independent FMAs added to the token
loop cost 0.20 cycles each against a 0.19 theoretical minimum, so there
is no idle slot left to fill. The machine is busy, not stalled. Speedups
have to come from issuing fewer instructions rather than scheduling them
better. Hardware `fdiv` and `fsqrt` beat Newton-refined reciprocal and
rsqrt by 2 to 4% for the same reason: the refinement costs five
instructions instead of one.

An fp16 MLP was tried and dropped. FMLAL, meaning fp16 multiply with
fp32 accumulate, halves the weight bytes but not the op count, and
changed nothing. Accumulating in fp16 as well does halve the op count and
gained 5%, at the cost of three orders of magnitude of logit accuracy
(max relative error 4e-05 against 3e-02).

Batching is the change that would lift the ceiling. Several independent
sequences at once turns every matvec into a matmul and gives the weights
the reuse they currently have none of.

## Performance

Single-threaded, on the benchmark's 5M token sampling loop:

| machine | backend | tok/sec | |
| --- | --- | --- | --- |
| Apple M5 Pro | NEON | 10,192,308 | median of 15 runs |
| AMD Ryzen | AVX2 | 6,733,185 | single run |

Both run the same algorithm and differ only in the kernels, but this is
not a fair comparison of the two instruction sets.

The M5 Pro sustains 5.27 FMAs and 5.75 128-bit loads per cycle, which is
a very wide core, and this workload is limited by issue width rather than
by stalls, so core width translates almost directly into throughput.

The AVX2 path is also untuned. Every constant in the NEON path, meaning
accumulator set counts, fused against unfused MLP, and tile sizes, was
chosen by A/B measurement on ARM hardware. None of that has been repeated
on x86. x86-64 also has 16 vector registers against ARM64's 32, so the
register-hungry choices are the ones most likely to be wrong there. There
is probably headroom in the AVX2 number.

For a while this figure was quoted from a Rosetta 2 run at 2,225,024
tok/sec. Native x86 turned out to be 3.0x that, because Rosetta
translates 256-bit AVX2 into pairs of 128-bit NEON operations and
discards the width the backend exists for. Emulation is fine for checking
correctness and useless for benchmarking.
