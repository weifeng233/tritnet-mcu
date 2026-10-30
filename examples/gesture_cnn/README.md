# gesture_cnn: a TinyML sensor classifier

The 1D-CNN task from EdgeImpulse and MLPerf-Tiny: four gesture classes (rest, shake,
circle, tap) from 3-axis accelerometer windows. The data is generated deterministically
from a fixed seed with random phase, amplitude jitter and per-sample noise, so the run
needs no external dataset. The model has 2,524 parameters and 54,208 MACs per inference:

```
input 3x128
Conv1D(3 -> 8, k=8)  + ReLU + MaxPool(2)   -> 8x60
Conv1D(8 -> 16, k=4) + ReLU + MaxPool(2)   -> 16x28
Flatten -> Dense(448 -> 4) + softmax
```

Biases stay fp32 (28 of 2,524 parameters); every conv kernel and the dense weight go
ternary. Each conv and pool kernel is tested against a direct reference, and the example
also runs a numerical gradient check end to end.

Measured on a Ryzen 9 7940H (4000 samples, 80/20 split, 2000 fp32 steps then 1500 QAT
steps, about 11 s total):

```
eval (accuracy, float activations)
  fp32  train 100.00%   val 100.00%   (val loss 0.0014)
  ptq   val 65.75%   (ternary, no QAT)
  qat   train 100.00%   val 100.00%
  trits: -1 42.6%   0 21.2%   +1 36.2%

storage
  packed 5-trits/byte  :    672 bytes  (15.0x smaller than fp32)

latency (host CPU, float activations)
  fp32 weights         :  45.67 us/inference  (1187.0M MACs/s)
  ternary weights      :  42.93 us/inference  (1262.6M MACs/s)

packed round-trip: OK (lossless)   qat recovery 99.9% of fp32->ptq loss gap
```

Naive ternarization drops val accuracy to 65.75 %; QAT brings it back to 100 % with every
kernel running as {-1, 0, +1} trits at 1.6 bits per weight.

Run from the repository root:

```bash
zig build --release=fast example-cnn    # writes examples/gesture_cnn/modeldata/cnn1d_packed.bin + cnn1d_valset.bin
make cnn-c                              # C deployment check (or: make deploy-check)
```

The C engine (`c/cnn1d_inference.c`) reads the packed artifact and the exported val
split, runs the network from the trits alone and recomputes the accuracy. The page in
`web/tinyml.html` charts the training run.
