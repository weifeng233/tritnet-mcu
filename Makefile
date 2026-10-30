# Default to zig's clang; `?=` is skipped because make's built-in default
# already defines CC, so the override is done explicitly when the user has
# not set CC themselves.
ifeq ($(origin CC),default)
CC := zig
endif

# `zig cc` is a drop-in clang. When CC is zig the subcommand has to be
# explicit, otherwise `zig -Ofast …` is treated as a zig build invocation.
ifeq ($(CC),zig)
  ZCC := zig cc
else
  ZCC := $(CC)
endif

# clang ignores the in-file `#pragma GCC target`, so the AVX2/FMA feature
# flags have to be passed explicitly or the intrinsics won't build.
ifeq ($(OS),Windows_NT)
  UNAME_M := x86_64
else
  UNAME_M := $(shell uname -m)
endif
ifeq ($(filter arm64 aarch64,$(UNAME_M)),)
  ARCHFLAGS ?= -march=native -mavx2 -mfma
else
  ARCHFLAGS ?= -march=native
endif

# zig's clang is more aggressive than gcc here: -Ofast (no strict IEEE, fast
# math, no signed-zero preservation) is what it takes to match gcc's
# -O3 -ffast-math and it comes out ~9% faster on this box. -ffast-math is
# implied by -Ofast, so it is not repeated.
CFLAGS  ?= -Ofast $(ARCHFLAGS) -Wall
LDLIBS  := -lm
BIN     := examples/gpt_names/microgpt-c
SRC     := examples/gpt_names/c/microgpt.c
DATA    := examples/gpt_names/data/names.txt

# C deployment engines: the framework's deploy leg. Each engine consumes the
# packed artifacts written by an example (`zig build --release=fast
# example-cnn` / `example-ncps`) and must reproduce the trainer's numbers
# exactly -- an independent second path over the same packed format.

# TinyML 1D-CNN deployment engine: consumes the artifacts written by
# `zig build --release=fast example-cnn` and verifies accuracy/latency parity.
CNN_BIN  := examples/gesture_cnn/cnn1d-c
CNN_SRC  := examples/gesture_cnn/c/cnn1d_inference.c
CNN_MODEL := examples/gesture_cnn/modeldata/cnn1d_packed.bin
CNN_VALSET := examples/gesture_cnn/modeldata/cnn1d_valset.bin

# Neural Circuit Policies deployment engine: consumes the artifacts written
# by `zig build --release=fast example-ncps` and verifies val-MSE parity.
NCPS_BIN  := ncps-c
NCPS_SRC  := c/ncps_inference.c

all: $(BIN)

$(BIN): $(SRC)
	$(ZCC) $(CFLAGS) -o $@ $< $(LDLIBS)

run: $(BIN)
	./$(BIN) $(DATA)

$(CNN_BIN): $(CNN_SRC)
	$(ZCC) $(CFLAGS) -o $@ $< $(LDLIBS)

cnn-c: $(CNN_BIN)
	./$(CNN_BIN) $(CNN_MODEL) $(CNN_VALSET)

$(NCPS_BIN): $(NCPS_SRC)
	$(ZCC) $(CFLAGS) -o $@ $< $(LDLIBS)

# Runs the deployment check on all four packed NCP artifacts.
.PHONY: ncps-check
ncps-check: $(NCPS_BIN)
	./$(NCPS_BIN) modeldata/ncps_cfc_packed.bin modeldata/ncps_valset.bin
	./$(NCPS_BIN) modeldata/ncps_fc_packed.bin modeldata/ncps_valset.bin
	./$(NCPS_BIN) modeldata/ncps_ncp_packed.bin modeldata/ncps_valset.bin
	./$(NCPS_BIN) modeldata/ncps_ltc_packed.bin modeldata/ncps_valset.bin

# Full deployment check: every packed artifact against its C engine.
.PHONY: deploy-check
deploy-check: cnn-c ncps-check

# Timestral-5M: the sketch-a-series app. `fetch-timestral` pulls the released
# checkpoint (21 MB) and refreshes the reference fixtures; `timestral` quantizes
# and scores it; `sketch` serves the drawing app on http://127.0.0.1:8000.
TIMESTRAL_CLI := zig-out/bin/timestral-cli
TIMESTRAL_SAFE := modeldata/timestral-5m/model.safetensors
TIMESTRAL_QAT := modeldata/timestral_qat/model.safetensors
TIMESTRAL_PACKED := modeldata/timestral_packed.bin
TIMESTRAL_PTQ := modeldata/timestral_ptq.bin

fetch-timestral:
	python -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Timestral-5M', local_dir='modeldata/timestral-5m', allow_patterns=['*.json','*.safetensors','LICENSE'])"
	python python/timestral_fixtures.py

$(TIMESTRAL_CLI): $(wildcard src/timestral/*.zig src/timestral.zig examples/timestral_sketch.zig)
	zig build --release=fast

# The deployed artifact: the QAT latents, ternarized, packed, width-calibrated.
quantize-timestral: $(TIMESTRAL_CLI) $(TIMESTRAL_QAT)
	$(TIMESTRAL_CLI) quantize $(TIMESTRAL_QAT) $(TIMESTRAL_PACKED)

# The baseline the report compares against: the released checkpoint, plain PTQ.
quantize-timestral-ptq: $(TIMESTRAL_CLI)
	$(TIMESTRAL_CLI) quantize $(TIMESTRAL_SAFE) $(TIMESTRAL_PTQ) --no-calibrate

timestral-eval: $(TIMESTRAL_CLI)
	$(TIMESTRAL_CLI) eval $(TIMESTRAL_SAFE) --dump modeldata/timestral_strokes.bin --json modeldata/timestral_eval_fp32.json
	$(TIMESTRAL_CLI) eval $(TIMESTRAL_PTQ) --json modeldata/timestral_eval_ptq.json
	$(TIMESTRAL_CLI) eval $(TIMESTRAL_PACKED) --json modeldata/timestral_eval_ternary.json
	$(TIMESTRAL_CLI) forecast $(TIMESTRAL_SAFE) modeldata/timestral_strokes.bin --out modeldata/timestral_forecast_fp32.bin
	$(TIMESTRAL_CLI) forecast $(TIMESTRAL_PACKED) modeldata/timestral_strokes.bin --out modeldata/timestral_forecast_ternary.bin

timestral-check: timestral-eval
	python python/timestral_parity.py
	python python/timestral_check_format.py $(TIMESTRAL_PACKED) $(TIMESTRAL_PTQ)

timestral-qat:
	python python/timestral_qat.py --steps 2500 --batch 32 --lr 1.5e-4

sketch: $(TIMESTRAL_CLI)
	python deploy/app.py --port 8000

clean:
	rm -rf $(BIN) $(CNN_BIN) $(NCPS_BIN) zig-out .zig-cache *.o *.dSYM

.PHONY: all run cnn-c ncps-check deploy-check clean fetch-timestral quantize-timestral quantize-timestral-ptq timestral-eval timestral-check timestral-qat sketch
