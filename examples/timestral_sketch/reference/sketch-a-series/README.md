---
title: Sketch-a-Series
emoji: ✏️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Draw a line. Timestral continues it as you draw.
models:
  - mistralai/Timestral-5M
---

# Sketch-a-Series

Draw a line from left to right, with a finger or a mouse. [Timestral-5M](https://huggingface.co/mistralai/Timestral-5M),
a time-series forecasting model from Mistral AI, continues it while you draw. The orange fan shows its forecast:
the line is the median, and the shaded bands are its quantiles from 5–95% to 45–55%.

**Fool it** mode: draw the start of a line and lift. The forecast freezes. Then draw how the line goes on. The score
is the share of your continuation that falls outside the forecast's 80% band.

## How it works

- The page resamples your stroke to one value per step (300 steps across the screen), so drawing speed does not
  change the series the model reads.
- Each time the stroke grows, the page sends it to the server, which forecasts every step to the right edge of
  the screen. Requests that arrive together are forecast in one batch.
- The model runs zero-shot on the CPU, with no training on drawings. A forecast takes about 10–20 ms.
- Timestral forecasts in 32-step patches. The page smooths each quantile over 7 steps so patch edges do not show
  as steps.

## Credits and licences

- Model: [Timestral-5M](https://huggingface.co/mistralai/Timestral-5M) by Mistral AI, released under a Modified
  MIT licence.
- Inference code: [mistralai/timestral](https://github.com/mistralai/timestral), Apache 2.0.
- Fonts: Archivo and IBM Plex Sans, SIL Open Font License.
