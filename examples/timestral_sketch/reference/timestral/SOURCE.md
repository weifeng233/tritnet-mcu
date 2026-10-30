# Vendored reference: timestral

`*.py` and `pyproject.toml` in this directory are a copy of the public package
<https://github.com/mistralai/timestral> (version 1.0.0 per `pyproject.toml`, fetched
2026-10-20), kept so that the Zig port can be diffed against the code it reproduces
and so that the TRFX fixtures can be regenerated.

License: Apache-2.0, reproduced in [`LICENSE`](LICENSE).

Files are unmodified except that `README.md` and packaging metadata that the
reference does not need were not copied. Changes, if any are ever made, must be
marked as Apache-2.0 §4(b) requires.

This tree is never compiled or linked into the Zig engine, and it is excluded from
GitHub's language statistics. `examples/timestral_sketch/python/timestral_fixtures.py` imports it to dump
reference tensors; `examples/timestral_sketch/deploy/app.py` imports it for the fp32 backend.
