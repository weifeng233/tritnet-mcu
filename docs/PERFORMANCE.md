# Performance notes

- **Applies to:** `main` as of v0.2.0
- **Last verified:** 2026-10-30
- **Verification:** `zig build --release=fast run` and `zig build --release=fast example-gpt` on the machine named below

How the inference forward differs from the training one, and where the
time per token goes.

## Inference path

Training and inference use separate forward passes. `gpt_forward` stores
~1.3 KB of activations per token for backprop. `gpt_forward_infer` needs
none of that and is built for one-token-at-a-time decoding. Its logits
match the training forward to 2.4e-06 absolute on AVX2, against logits
of magnitude ~8, which is fp32 rounding noise.

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

About half the time is the MLP.

The limit is issue width rather than stalls. Extra independent FMAs
added to the token loop cost 0.20 cycles each against a 0.19 theoretical
minimum, so there is no idle slot left to fill: the machine is busy, not
waiting. Speedups therefore have to come from issuing fewer instructions
rather than from scheduling them better.

Single-threaded, on the benchmark's 5M token sampling loop:

| machine | backend | tok/sec | |
| --- | --- | --- | --- |
| AMD Ryzen 9 7940H | AVX2 | 5,491,565 | median of 5 runs |

## Compiler

The Zig build (`zig build --release=fast`) is `ReleaseFast` (`-Ofast` semantics).
On the AMD Ryzen 9 7940H the two zig-build binaries match within
run-to-run noise: 5.49M (`zig fp32+AVX2`) against 5.39M (`c fp32+AVX2`)
tok/sec, medians of 5 runs on the 5M-token loop. The Makefile still
defaults to `zig cc`, Zig's drop-in clang; built through the Makefile the
same C source measures 5.81M, while gcc at `-Ofast` measures 5.23M —
zig's clang is roughly 11% faster (medians of 5 runs each). `make
CC=gcc` reproduces the gcc numbers. Session-to-session drift on this box
is a few percent, so compare numbers from the same measurement session.

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
  to 8 measured flat on this machine.
- Breaking the token-to-token dependency, tested by replaying a recorded
  token trajectory so consecutive tokens were independent. No faster, so
  nothing is waiting on that dependency.

Batching is the change that would lift the ceiling. Several independent
sequences at once turns every matvec into a matmul and gives the weights
the reuse they currently have none of.

## Training path

The training loop (`gpt_forward`/`gpt_backward`/Adam) was scalar for a
long time; strict FP evaluation kept LLVM from auto-vectorizing the
reductions and accumulation loops. It now uses the same `@Vector` style
as the inference path, with 8-wide kernels on x86 and NEON:

- `linear_bwd_x` walks rows sequentially and keeps one 8-wide
  accumulator set per weight column, instead of striding down `nout`
  rows per output channel. Per-column sums stay in row order, so values
  match the scalar version.
- `linear_bwd_w` is an 8-wide FMA outer product per row.
- `adam_update` is 8-wide and takes the bias corrections and the clip
  factor as arguments. The gradient clip is folded into Adam's gradient
  read (`g * scale`) instead of a second full pass that rescales and
  rewrites every gradient array.
- The gradient-norm reduction is one fused-multiply pass over each
  gradient array.
- Attention forward/backward accumulate the 4-wide head slices with
  `Vec4` FMAs against the kv rows; residual adds, the ReLU² and the
  embedding/scatter paths are 8-wide.
- Per-step fixed costs: the kv-cache reset before forward was dead
  (each position writes k/v before attention reads them), the dk/dv
  accumulator reset only covers the `n` live positions, Adam's
  `pow(b1, step)` bias correction is computed once per step, and
  `char_to_id` is a 256-entry LUT.

ReleaseFast on the 7940H, 20000 steps over `data/names.txt` (median of
5 runs): 0.30 s, roughly 67K steps/sec. Final average loss 2.2194 vs
2.2199 scalar — the only
numerical changes are reassociated sums (vector lane splits, FMA
contraction in the new kernels), and samples remain name-like.
Inference throughput is untouched, as none of its code paths changed.
The training timer now prints `train: <s> (<steps/sec>)` after the
last step.
