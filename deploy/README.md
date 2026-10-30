# Sketch-a-Series, deployed locally

The app half of this repository's Timestral example: draw a stroke with the mouse, and
Timestral-5M continues it as a fan of quantiles. It mirrors
[sketch-a-series](https://huggingface.co/spaces/townbox/sketch-a-series) on
[timestral](https://huggingface.co/mistralai/Timestral-5M), and it can draw the same
stroke through two backends at once:

| backend | what runs | weights |
| --- | --- | --- |
| `fp32` | Mistral's released checkpoint, PyTorch (reference `timestral` package) | 20.8 MB |
| `ternary` | this repository's Zig engine on the packed artifact (absmean QAT + 5 trits/byte) | 1.15 MB |

Both answer the identical `/api/forecast` request, so the page draws the fp32 median as a
dashed line and the ternary median as a solid line on the same chart.

## Run it

```bash
# 1. the checkpoint (21 MB) + the reference fixtures used by `zig build test`
make fetch-timestral

# 2. the Zig worker + quantizer
zig build --release=fast

# 3. the ternary artifact from the released checkpoint
./zig-out/bin/timestral-cli quantize modeldata/timestral-5m/model.safetensors modeldata/timestral_packed.bin

# 4. serve
python deploy/app.py                  # http://127.0.0.1:8000
python deploy/app.py --backends ternary   # no PyTorch in the loop at all
```

`python deploy/app.py --help` lists the flags; `TIMESTRAL_WORKER`, `TIMESTRAL_ARTIFACT`
and `TIMESTRAL_FP32` override the paths.

## How a request flows

```
browser (canvas)
  │  POST /api/forecast {"y": [...], "h": 96, "backend": "both"}
  ▼
deploy/app.py  (FastAPI)
  ├─ fp32    → timestral.TimestralForecaster.forecast       (PyTorch, in-process)
  └─ ternary → timestral-cli serve  (subprocess, one line of JSON per request)
                 └─ artifact.bin → trits → decodeBlock → refine frame → clampsym
```

The worker holds the artifact in memory: no per-request load, no PyTorch import, and the
ternary matmul is add/skip/subtract over the packed trits with one scale per output row.

## API

`GET /api/info`

```json
{"model": "Timestral-5M", "levels": [0.01, "...", 0.99], "h_values": [96, 192],
 "backends": [{"id": "fp32", "params": 5205504, "bytes": 20843960},
              {"id": "ternary", "params": 5205504, "bytes": 1152070, "bits_per_weight": 1.77}],
 "eval": {"fp32": {...}, "ternary": {...}}}
```

`POST /api/forecast` — `{"y": [floats], "h": 96, "backend": "both" | "fp32" | "ternary"}`
→ `{"h": 96, "levels": [...], "results": {"fp32": {"q": [[21 floats] ... 96], "ms": 10.4},
"ternary": {"q": [...], "ms": 13.2}}}`. `q[i][k]` is the quantile at `levels[k]` for
horizon step `i`; level index 10 is the median. If one backend fails the other still
answers (the response carries an `errors` object).

## Notes

- The page resamples the ink to a uniform grid before sending (the model wants consecutive
  time steps, not pixel positions), throttles live requests to ~120 ms, and supersedes
  in-flight ones with an `AbortController`.
- `h` is one decode block of patches: 96 and 192 steps here, well inside the model's 1024.
- The fp32 backend needs `torch` and the reference package in `reference/timestral`
  (`sys.path` is set up by `app.py`); the ternary backend needs neither.
