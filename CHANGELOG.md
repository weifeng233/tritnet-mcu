# Changelog

All notable changes to this repository. Numbers are the ones the committed artifacts
and eval records report; if a figure here disagrees with a document, the document and
this file are both wrong and should be fixed together.

## v0.2.0 — 2026-10-30

The standardization release: the repository is now published, licensed correctly, and
every claim in it points at something reproducible.

### Added

- **Timestral-5M deployment** (`src/timestral/`, `examples/timestral_sketch.zig`): a
  stage-by-stage port of Mistral's reference implementation, a `TRTL` v2 packed
  artifact of 1.15 MB against 21 MB of fp32 weights, and a `timestral-cli` with
  `quantize` / `eval` / `forecast` / `serve`.
- **Quantization-aware fine-tuning** (`python/timestral_qat.py`) plus the evaluation
  tooling around it: sensitivity per layer group, mixed-precision study, artifact
  checker, parity harness, width calibration, figures.
- **sketch-a-series app** (`deploy/app.py`, `web/sketch-a-series.html`) answering one
  endpoint from both the fp32 reference and the ternary worker, so the cost of
  quantization is visible per stroke rather than quoted.
- **`docs/ARTIFACT-FORMAT.md`**: the five binary formats, the ternary codec, the
  golden vectors, and the incompatibility policy.
- **`docs/OWNERSHIP.md`** and **five ADRs** (`docs/adr/`): what is original, what is a
  port, what is vendored — and why each big choice was made.
- **`pyproject.toml` + `uv.lock`**: the Python side has a lock file now.
- **`.gitattributes`**: LF everywhere, and vendored trees excluded from the language
  stats so the repository is reported as Zig.

### Fixed

- `zig build test` no longer needs `zig-out/` to exist: the artifact round-trip test
  writes to a temporary directory, so a clean clone passes.
- The parity tests no longer skip silently. Without the 21 MB checkpoint they report
  what they skipped; with `-Drequire-parity=true` (what CI uses) a missing model is a
  failure instead of a green build that tested nothing.
- `python/export.py` imported `bitnet_gpt_train`, which does not exist; it now imports
  `train_qat` and regenerates `c/model.h` byte for byte from the committed checkpoint.
- `weights.load` reports `FileNotFound` rather than folding every read error into
  `Truncated`, so "not downloaded" and "corrupt" are different messages.

### Changed

- **`LICENSE`** now names this repository's author instead of the author of the
  upstream microGPT-C, and points at `THIRD_PARTY.md`. **`THIRD_PARTY.md`** is new and
  records every upstream, its license, and whether it is vendored or merely credited.
- **`reference/BitNetMCU`** (GPL-3.0) and **`reference/mlp-mcu`** (unlicensed early
  prototype) are gone from the tree; history was rewritten before publishing so the
  GPL tree never appears in the public repository. `reference/timestral` now carries
  its Apache-2.0 `LICENSE`, and `reference/sketch-a-series` says plainly that it has no
  license of its own.
- **Generated artifacts left git**: the 2.3 MB of packed/ptq/forecast binaries are
  rebuilt by `make quantize-timestral` and published as release assets. Fixtures and
  eval records stay committed, because a clean clone has to be able to test itself.

## v0.1.0 — 2026-10-30

First tagged snapshot: the Timestral deployment and the ternary quantization work, as
it stood before standardization. Kept so that the cleanup above is a diff and not a
rewrite of history.
