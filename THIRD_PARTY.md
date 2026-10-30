# Third-party components

This repository's own source is MIT (see [`LICENSE`](LICENSE)). It also contains
code derived from, or vendored copies of, the projects below. Each stays under its
own license, and nothing here grants you rights to those projects under MIT.

BitNetMCU and microGPT-C shaped the design without contributing code to it.

| Component | Upstream | License | Where it is | How it is used |
| --- | --- | --- | --- | --- |
| microGPT-C | <https://github.com/vixhal-baraiya/microgpt-c> | MIT | `examples/gpt_names/c/microgpt.c` (vendored), `examples/gpt_names/fp32.zig` (Zig port) | The fp32 baseline and the A/B twin of the Zig trainer. The names example trains the same 4,192-parameter char-level transformer. |
| BitNetMCU | <https://github.com/cpldcpu/BitNetMCU> | GPL-3.0 | not in this repository | Primary inspiration: the absmean ternary scale, the straight-through estimator, and the idea of deploying packed trits to an MCU. No BitNetMCU source is vendored, compiled or linked here. |
| mlp-mcu | this repository's own earlier prototype | MIT (this repository) | removed from the current tree, still in history | The first attempt: BitNetMCU's layer structure over a microGPT-C-style model, in plain C. Superseded by the Zig framework, kept in history for provenance. |
| timestral | <https://github.com/mistralai/timestral> | Apache-2.0 | `examples/timestral_sketch/reference/timestral/` | The PyTorch reference implementation of Timestral. Used to generate the TRFX parity fixtures and as the fp32 backend of the sketch app. Never compiled into the Zig engine. |
| ncps | <https://github.com/mlech26l/ncps> | Apache-2.0 | not vendored; ported in `examples/liquid_sine/liquid_sine.zig` and `examples/liquid_sine/c/ncps_inference.c` | The NCP wiring generator (`wirings.NCP` / `AutoNCP`, `FullyConnected`), the CfC cell in all three modes, and the `torch_cfc_sinusoidal.py` task, ported to Zig and to C. What is ported, and every deliberate deviation (wiring RNG, init scheme), is listed in the header of `examples/liquid_sine/liquid_sine.zig` as Apache-2.0 §4(b) requires. |
| sketch-a-series | <https://huggingface.co/spaces/townbox/sketch-a-series> | the Space's own license | `examples/timestral_sketch/reference/sketch-a-series/` | UX reference only. `examples/timestral_sketch/deploy/app.py` and `examples/timestral_sketch/web/sketch-a-series.html` are independent implementations. |
| Timestral-5M weights | <https://huggingface.co/mistralai/Timestral-5M> | Modified MIT | downloaded to `examples/timestral_sketch/modeldata/timestral-5m/`, never committed | Evaluated and quantized. Fetch with `make fetch-timestral`. |
| 5-trits-per-byte packing | <https://compilade.net/blog/ternary-packing> | a blog post, not code | `src/framework/ternary.zig` | The base-3 encoding with ceiling rescale and the division-free decode. Reimplemented from the description, credited in the module docstring. |

## Why BitNetMCU is not vendored

BitNetMCU is GPL-3.0. Reading it for ideas does not make this repository GPL, but
redistributing its source inside an MIT repository would, and "this repository is
MIT" would stop being true. It was removed from the tree and from history rather
than quarantined under a `reference/` tree, so that claim stays clean.

The absmean ternarization recipe, the scale/STE pairing and the fact that a ternary
network can run on an MCU are the whole debt.

## Vendored trees

`examples/timestral_sketch/reference/timestral/` is a copy of the upstream package at
the version and date noted in
[`examples/timestral_sketch/reference/timestral/SOURCE.md`](examples/timestral_sketch/reference/timestral/SOURCE.md),
with its Apache-2.0 `LICENSE` alongside.
`examples/timestral_sketch/reference/**` is excluded from GitHub's language
statistics ([`.gitattributes`](.gitattributes)) and is never part of a build.

## Ports, which are not vendoring

Two examples are ports rather than original designs:
`examples/liquid_sine/liquid_sine.zig` ports the ncps CfC/NCP stack, and
`examples/gpt_names/fp32.zig` ports microGPT-C. Both keep the upstream license terms,
credit the source in their header comment, and record where behavior differs from the
original (different RNG stream, different weight init). Those two file headers answer
the question of authorship; this table does not.

## Generated artifacts under third-party terms

`examples/timestral_sketch/modeldata/fixtures/timestral_fixture.bin` holds intermediate
tensors produced by running `examples/timestral_sketch/reference/timestral` on the
Timestral-5M weights, so it is derived from both: treat it as Apache-2.0 code output
over Modified-MIT weights. It exists so the Zig port can be checked against PyTorch in
a clean clone without downloading a 21 MB checkpoint. The packed artifacts
(`examples/timestral_sketch/modeldata/timestral_packed.bin` and friends) are produced
entirely by this repository's code and are published as release assets rather than
committed.

## Reporting a mistake

If a file here is mis-attributed or used beyond its license, open an issue with the
path and the upstream it came from. An attribution error is a bug.
