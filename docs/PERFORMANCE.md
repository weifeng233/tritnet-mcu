# Performance notes

How the inference forward differs from the training one, and where the
time per token goes.

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
  chain of `nin` FMAs, so `mv16_blk16r` uses two sets and
  `mv16_blk16r4`, which fc1 calls, uses four.
- Hoisted prefix. The embedding, both RMS norms and layer 0 Q/K/V depend
  only on `(token, pos)`, and there are only `vocab * BLOCK_SIZE`
  distinct inputs, so `build_pretok()` puts them in a table.
- Deferred RMS scale. The MLP norm scale is a positive scalar and
  fc1/ReLU^2/fc2 are positively homogeneous, so `rmsnorm_scale()`
  returns it and it factors out to one `s^2` at the end, off fc1's
  critical path. An activation that is not positively homogeneous would
  break this.
- Fused sampling. `sample_logits()` does softmax, temperature and
  weighted choice in a single pass over unnormalised weights.
- Vocab padded to 4 rather than 16. `mv16_blk16`, `mv16_blk8` and
  `mv16_blk4` emit 16/8/4-row tiles, so a 27-token vocab costs 28 rows
  instead of 32.

## Where the time goes

About half the time is the MLP, roughly 240 of 500 cycles per token.

The limit is issue width rather than stalls. Extra independent FMAs
added to the token loop cost 0.20 cycles each against a 0.19 theoretical
minimum, so there is no idle slot left to fill: the machine is busy, not
waiting. Speedups therefore have to come from issuing fewer instructions
rather than from scheduling them better.

Single-threaded, on the benchmark's 5M token sampling loop:

| machine | backend | tok/sec | |
| --- | --- | --- | --- |
| Apple M5 Pro | NEON | 10,168,430 | median of 15 runs |
| AMD Ryzen 5 5600H | AVX2 | 6,927,775 | median of 5 runs |

Both run the same algorithm and differ only in the kernels, so this is
not a fair comparison of the two instruction sets. The M5 Pro sustains
5.27 FMAs and 5.75 128-bit loads per cycle, which is a wide core, and on
an issue-width-limited workload that translates almost directly into
throughput.

The AVX2 row is native x86. The same binary under Rosetta 2 on Apple
Silicon measures 2,225,024 tok/sec, a third of native, because Rosetta
translates 256-bit AVX2 into pairs of 128-bit NEON operations and
discards the width the backend exists for. Emulation is usable for
checking correctness and not for benchmarking.

The AVX2 path is the less tuned of the two. Every constant in the NEON
path was chosen by A/B measurement on ARM hardware and almost none of
that has been repeated on x86. Note that an accumulator set holds 16
floats, so it is 4 registers and 4 independent chains on NEON but only 2
of each on AVX2: the same source gives x86 half the chains. x86-64 also
has 16 vector registers against ARM64's 32, which makes the
register-hungry choices the likeliest place to find headroom.

## Dead ends

Measured and abandoned, so they need not be tried again:

- fp16 weights. FMLAL, meaning fp16 multiply with fp32 accumulate,
  halves the weight bytes but not the op count and moved the MLP by one
  cycle out of 241. Accumulating in fp16 as well does halve the op count
  and gained 5%, at the cost of three orders of magnitude of logit
  accuracy.
- Newton-refined reciprocal and rsqrt, 2 to 4% slower than hardware
  `fdiv` and `fsqrt`, because the refinement is five instructions where
  the hardware is one.
- More accumulator chains on AVX2. Raising Wo and lm_head from 4 chains
  to 8 measured flat on a 5600H.
- Breaking the token-to-token dependency, tested by replaying a recorded
  token trajectory so consecutive tokens were independent. No faster, so
  nothing is waiting on that dependency.

Batching is the change that would lift the ceiling. Several independent
sequences at once turns every matvec into a matmul and gives the weights
the reuse they currently have none of.
