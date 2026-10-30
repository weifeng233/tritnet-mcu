# Changelog

All notable changes to this repository. Numbers come from the committed artifacts and
eval records. If a figure here disagrees with a document, both are wrong and should be
fixed together.

## v0.3.0 (2026-10-30)

### Changed

- One directory per example. The models no longer share `c/`, `python/`, `data/`,
  `modeldata/`, `deploy/`, `reference/` or `web/` trees. Each example owns its code,
  data and artifacts under `examples/<model>/` (`gpt_names/`, `gesture_cnn/`,
  `liquid_sine/`, `timestral_sketch/`), and `src/` holds only the framework
  (`src/framework.zig` plus `src/framework/`). `src/fp32.zig` became
  `examples/gpt_names/fp32.zig`, `src/timestral*` became
  `examples/timestral_sketch/timestral*`, and every example now writes to its own
  `modeldata/`. Build steps, make targets, CI jobs and document paths follow, and the
  21 MB checkpoint and the packed artifacts stay gitignored at their new paths.
- The decision records and the research notes are no longer published. `docs/adr/`
  and `docs/research/` are listed in `.gitignore` and stay on disk as private notes,
  and both were removed from the published history. The reasoning they carried is now
  summarized, conclusions only, in the README's design decisions section.
- `docs/TIMESTRAL.md` no longer overstates what quantization cost. The QAT latents
  score 0.0946 without quantization, not 0.0769 (+33 % over the released checkpoint,
  not ~8 %), and the page's 31-44 → 16-22 ms gap is PyTorch against the Zig engine: the
  same engine is 3-8 % slower on trits, so 1.77 bits buys compression, not speed. The
  note that measured both is private now, so both corrections live in
  `docs/TIMESTRAL.md` itself.
- The README's example-2 numbers now come from the 2026-10-30 run: 42.93 µs per
  inference (1262.6 M MACs/s) against 45.67 µs for fp32 weights on the same host,
  replacing an older 41.1 µs figure that no committed record produced.
- Every published document was rewritten to remove the habits of generated text:
  staged openers, not-X-but-Y formulas, one-line closers, decorative bold, and dashes
  used as all-purpose connectors. The claims, numbers and paths are unchanged.

### Removed

- `docs/OWNERSHIP.md`: the file-level ownership map. Licensing provenance now lives in
  `THIRD_PARTY.md`.
- `docs/cnn_run.txt`: a captured CNN run log. `web/tinyml.html` records its own capture
  instead of pointing at it.
- `docs/PERFORMANCE.zh-CN.md`: a translation of `docs/PERFORMANCE.md` that had to be
  maintained twice.
- `.github/release-notes-v0.2.0.md`: a copy of the release body GitHub already holds.
- `docs/app-screenshot.png`: a screenshot nothing referenced.

## v0.2.0 (2026-10-30)

The standardization release: the repository became public, the licensing was made
accurate, and every claim in it points at something reproducible.

### Added

- Timestral-5M deployment (`src/timestral/`, `examples/timestral_sketch.zig`): a
  stage-by-stage port of Mistral's reference implementation, a `TRTL` v2 packed
  artifact of 1.15 MB against 21 MB of fp32 weights, and a `timestral-cli` with
  `quantize`, `eval`, `forecast` and `serve`.
- Quantization-aware fine-tuning (`python/timestral_qat.py`) plus the evaluation
  tooling around it: sensitivity per layer group, mixed-precision study, artifact
  checker, parity harness, width calibration, figures.
- the sketch-a-series app (`deploy/app.py`, `web/sketch-a-series.html`) answering one
  endpoint from both the fp32 reference and the ternary worker, so the cost of
  quantization is visible per stroke rather than quoted.
- `docs/ARTIFACT-FORMAT.md`: the five binary formats, the ternary codec, the golden
  vectors, and the incompatibility policy.
- `docs/OWNERSHIP.md` and five ADRs (`docs/adr/`): what is original, what is a port,
  what is vendored, and why each big choice was made.
- `pyproject.toml` and `uv.lock`: the Python side has a lock file now.
- `.gitattributes`: LF everywhere, and vendored trees excluded from the language stats
  so the repository is reported as Zig.

### Fixed

- `zig build test` no longer needs `zig-out/` to exist: the artifact round-trip test
  writes to a temporary directory, so a clean clone passes.
- The parity tests no longer skip silently. Without the 21 MB checkpoint they report
  what they skipped; with `-Drequire-parity=true` (what CI uses) a missing model is a
  failure instead of a green build that tested nothing.
- `python/export.py` imported `bitnet_gpt_train`, which does not exist; it now imports
  `train_qat`, and `SOURCE_DATE_EPOCH` pins the header timestamp so two runs can be
  compared with a plain `diff`. With the epoch set, the regenerated `c/model.h` matches
  the committed one except for that line.
- `weights.load` reports `FileNotFound` rather than folding every read error into
  `Truncated`, so "not downloaded" and "corrupt" are different messages.
- The CNN artifact (`CNN1Q002`) gained the footer the ADRs claimed it already had: the
  trainer's val accuracy, which `c/cnn1d_inference.c` recomputes from the trits and
  compares. Corrupting the artifact now exits non-zero instead of printing 24.62% and
  returning 0.
- `TRTL` rejects trailing bytes, `TRFX` validates its version byte, and the fixture
  reader bounds-checks every length field: a truncated fixture is an error, not a panic.

### Changed

- `LICENSE` now names this repository's author instead of the author of the upstream
  microGPT-C, and points at `THIRD_PARTY.md`, which is new and records every upstream,
  its license, and whether it is vendored or merely credited.
- `reference/BitNetMCU` (GPL-3.0) is gone from the tree, and history was rewritten
  before publication so it never appears in the public repository.
  `reference/mlp-mcu` is also gone from the tree, but it is this author's own earlier
  prototype (see `THIRD_PARTY.md`), kept in history for provenance; it was removed
  because the Zig framework supersedes it, not for licensing reasons.
  `reference/timestral` now carries its Apache-2.0 `LICENSE`, and
  `reference/sketch-a-series` says plainly that it has no license of its own.
- Generated artifacts left git: the 2.3 MB of packed, ptq and forecast binaries are
  rebuilt by `make quantize-timestral` and published as release assets. Fixtures and
  eval records stay committed, because a clean clone has to be able to test itself.
- Throughput numbers now carry their conditions. The old README quoted 5,491,565
  tok/sec without saying which power plan produced it; on 2026-10-30 the same binaries
  measured 8.44 M on 高性能 and about 4.0 M on 平衡. The condition is now part of the
  table, because the plan is worth more than every optimization in this repository
  combined.

## Pre-standardization snapshot (2026-10-30)

The Timestral deployment and ternary quantization work as it stood before the cleanup
above. It is not published as a tag: at that point `LICENSE` still carried the inherited
microGPT-C copyright line, `THIRD_PARTY.md` did not exist, and `reference/mlp-mcu` had
no license header, so the snapshot did not meet its own licensing claim.

The commit is still an ancestor of `main` (reachable with `git log`), and the local tag
`v0.1.0-pre-standardize` points at it for rollback. Commits before the licensing change
carry the inherited microGPT-C copyright line; that is a record of history, not a
statement about the current license.
