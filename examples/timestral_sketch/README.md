# timestral_sketch: Timestral-5M, deployed and ternarized

This example deploys a model trained elsewhere: Mistral AI's
[Timestral-5M](https://huggingface.co/mistralai/Timestral-5M) (5.2 M parameters, 8
patch-transformer blocks, 21 quantiles per step). It stands the model up behind a
*sketch-a-series* app: draw a stroke and the model continues it as a quantile fan, the
interaction of
[the Space](https://huggingface.co/spaces/townbox/sketch-a-series). Then it quantizes
that model with this framework's own stack (absmean QAT, 5 trits per byte, a packed
artifact, a Zig engine) and runs the same page against both backends, so the cost of 1.77
bits per weight is visible side by side.

The Zig engine is a full port of the forward pass: per-patch causal normalization,
`[Enc ; Dec]` attention with RoPE and sigmoid gates, SwiGLU, the monotone quantile head,
the advancing refine frame, and clampsym. It reproduces PyTorch to 2.4·10⁻⁶ relative error
on the fp32 weights, so the gap that follows is quantization and not a porting bug.

```
fp32 (safetensors, PyTorch)      20,843,960 B
ternary artifact (Zig, 1.77 bit)  1,152,070 B   (18.1x smaller)

24 held-out strokes, 256-step context, 96-step horizon
  variant      pinball   MAE     MASE    cov90   one request
  fp32          0.0711   0.217   5.89    0.961     31-44 ms
  PTQ ternary   0.2023   0.522  17.55    0.777        -
  QAT ternary   0.1071   0.293   8.91    0.901     16-22 ms
```

Naive ternarization triples the error. 2,500 steps of quantization-aware fine-tuning
(9.5 min on four CPU threads, `python/timestral_qat.py`) recover 73 % of that gap. What
survives best is what a sketch mostly is: trends and noisy walks come out within 10 % of
fp32, while strokes whose continuation needs an amplitude, such as drawn curves and
sinusoids, come out visibly damped at 2.2-2.8× the pinball. The app draws both medians on
the same chart so the cost is visible.

Run from the repository root:

```bash
make fetch-timestral                 # the released checkpoint (21 MB)
make timestral-qat quantize-timestral   # fine-tune, then quantize + calibrate
make timestral-check                 # parity against PyTorch on the same strokes
make sketch                          # http://127.0.0.1:8000, both backends
```

The app lives in `deploy/` (run instructions and the API contract in `deploy/README.md`),
the page in `web/sketch-a-series.html`, and the full write-up in
[docs/TIMESTRAL.md](../../docs/TIMESTRAL.md).
