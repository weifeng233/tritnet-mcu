CC ?= cc

# clang ignores the in-file `#pragma GCC target`, so on x86 the AVX2/FMA
# feature flags have to be passed explicitly or the intrinsics won't build.
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

CFLAGS  ?= -O3 $(ARCHFLAGS) -ffast-math -Wall
LDLIBS  := -lm
BIN     := microgpt
SRC     := src/microgpt.c
DATA    := data/names.txt

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

run: $(BIN)
	./$(BIN) $(DATA)

clean:
	rm -rf $(BIN) *.o *.dSYM

.PHONY: all run clean
