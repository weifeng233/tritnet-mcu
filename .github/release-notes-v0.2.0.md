# v0.2.0 — standardized, licensed, published

The Timestral-5M deployment and ternary quantization work, cleaned up to the point
where the repository answers five questions without a conversation: can I reproduce
it, is it tested, what is mine, can I keep changing it, and can I explain it.

## Highlights

- **Timestral-5M on a 1.15 MB artifact.** A stage-by-stage Zig port of the reference
  forecaster (worst relative difference 1.35e-5 across 13 stages), ternarized with
  absmean QAT and packed at 5 trits/byte: 21 MB → 1.15 MB, 1.77 bits/weight.
- **A second implementation of the same numbers.** `c/cnn1d_inference.c` and
  `c/ncps_inference.c` recompute each metric from the packed trits alone and compare
  against the value the trainer wrote into the artifact's footer.
- **Formats are specified** in [`docs/ARTIFACT-FORMAT.md`](docs/ARTIFACT-FORMAT.md),
  including four golden vectors for the ternary codec — enough to write an independent
  reader and check it.
- **Licensing is honest.** `LICENSE` names this repository's author; `THIRD_PARTY.md`
  lists every upstream, its license, and whether it is vendored or only credited.
- **Tests fail loudly.** `-Drequire-parity=true` turns a missing checkpoint into a red
  build instead of a skip; the round-trip test no longer needs `zig-out/` to exist.

## Artifacts

Rebuild any of these with `make quantize-timestral` / `make timestral-eval`; the
`model.safetensors` they derive from is not committed (fetch it with
`make fetch-timestral`).

| asset | bytes | sha256 |
| --- | --- | --- |
| `timestral_packed.bin` | 1,152,070 | `6dfd790805e84a3b41e7ab1bf85e14d60be2a44eb6cdff9f4f4525db1664055e` |
| `timestral_ptq.bin` | 1,152,070 | `801ea1a6fe83d89a3a717209194f148577b9817f619bb9abf8125ed401f15965` |
| `timestral_forecast_fp32.bin` | 193,536 | `c2092bd718481bb25ca9939247df3eae2d299a17ccffc3aa7f220d59b3010728` |
| `timestral_forecast_ternary.bin` | 193,536 | `ace3830cbc5a8c0c96d3cad8ff07495d8548cb13985dc2e05b47dd4814a7cf61` |

`packed` is the QAT artifact the app serves; `ptq` is the plain post-training
quantization baseline it is compared against (same size, worse metrics — that
difference is the point of `python/timestral_qat.py`). The two forecasts are the same
24 strokes through both backends, little-endian `f32[24][96][21]`.

```bash
zig build --release=fast
zig-out/bin/timestral-cli eval timestral_packed.bin --json eval.json
```

## Upgrading

The packed format is `TRTL` v2; a v1 file is rejected with `BadVersion` rather than
guessed at. Anyone who was on the pre-standardization snapshot (`v0.1.0`) should note
that history was rewritten before publication to drop the vendored GPL tree, so old
commit hashes no longer resolve.
