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

all: $(BIN)

$(BIN): $(SRC)
	$(ZCC) $(CFLAGS) -o $@ $< $(LDLIBS)

run: $(BIN)
	./$(BIN) $(DATA)

clean:
	rm -rf $(BIN) zig-out .zig-cache *.o *.dSYM

.PHONY: all run clean