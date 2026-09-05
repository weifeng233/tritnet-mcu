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
BIN     := microgpt-c
SRC     := c/microgpt.c
DATA    := data/names.txt

# C deployment engines: the framework's deploy leg. Each engine consumes the
# packed artifacts written by an example (`zig build --release=fast
# example-cnn` / `example-ncps`) and must reproduce the trainer's numbers
# exactly -- an independent second path over the same packed format.

# TinyML 1D-CNN deployment engine: consumes the artifacts written by
# `zig build --release=fast example-cnn` and verifies accuracy/latency parity.
CNN_BIN  := cnn1d-c
CNN_SRC  := c/cnn1d_inference.c
CNN_MODEL := modeldata/cnn1d_packed.bin
CNN_VALSET := modeldata/cnn1d_valset.bin

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

clean:
	rm -rf $(BIN) $(CNN_BIN) $(NCPS_BIN) zig-out .zig-cache *.o *.dSYM

.PHONY: all run cnn-c ncps-check deploy-check clean