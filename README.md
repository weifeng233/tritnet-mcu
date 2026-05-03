# MicroGPT-C
The most atomic way to train and inference a GPT in pure, dependency-free C.

## Compilation and Running

Compile with optimizations:

```bash
gcc -O3 -march=native -ffast-math -o microgpt microgpt.c -lm
```

**Flags explained:**
- `-O3`: Maximum optimization
- `-march=native`: Use CPU-specific instructions (AVX, etc.)
- `-ffast-math`: Faster floating point (trades some precision)
- `-lm`: Link math library (for sqrt, exp, etc.)

Run:

```bash
./microgpt
```

You should see loss decreasing, then generated samples!

```bash
step     1 / 20000 | loss 3.3178  (avg 3.3002)
step  5000 / 20000 | loss 2.6062  (avg 2.2945)
step 10000 / 20000 | loss 1.9652  (avg 2.2562)
step 15000 / 20000 | loss 2.6978  (avg 2.2153)
step 20000 / 20000 | loss 2.3462  (avg 2.2188)

--- inference ---
sample  1: kayley
sample  2: maria
sample  3: arana
sample  4: shayan
sample  5: jayden
sample  6: arian
sample  7: saria
sample  8: amari
sample  9: alila
sample 10: mailyn
  c fp32+AVX2               2,631,689 tok/sec
```

