# liquid_sine: Neural Circuit Policies (CfC/LTC)

A port of the [ncps](https://github.com/mlech26l/ncps) stack (Lechner & Hasani) onto the
framework: continuous-time RNNs whose sparse wiring forms a circuit of sensory, inter,
command and motor neurons. Implemented from the reference:

- `wirings.NCP` / `AutoNCP`: the stochastic 3-layer circuit constructor (sensory fanout,
  inter fanout, recurrent command synapses, motor fanin, ±1 polarities, reachability
  repair), plus `wirings.FullyConnected`
- `torch.CfCCell` in all three modes (default / pure / no_gate) with per-step timespans,
  and the `WiredCfCCell` layer stack whose ff1/ff2 projections are masked by the wiring
  adjacency; the time-gate projections stay dense, as in ncps
- `torch.LTCCell`: the original liquid time-constant ODE, solved with 6 Euler fixed-point
  unfolds per step. BPTT runs the chain rule back through every unfold, and a
  finite-difference check covers it.
- the `torch_cfc_sinusoidal.py` task (sin/cos in, double-frequency sine out, MSE),
  generalized into a 1,600-sequence dataset with random phase and jittered frequency

Each sequence is 48 steps of [sin, cos] over [0, 3·pi·k], and the target is the same
waveform at twice the frequency. Two details keep the wiring intact through
quantization: the sparsity is baked into the trits themselves (masked synapses have zero
latents and therefore zero trits, so the packed artifact needs no side-car mask), and the
absmean scale covers *wired* synapses only, so the zeros of the sparse circuit never
dilute it. Four models train in one run, as in the ncps sinusoidal example:

| model | params | fp32 val MSE | ptq val | qat val | qat recovery | packed |
| --- | --- | --- | --- | --- | --- | --- |
| CfC(2→32, backbone 64, proj 1) | 14,689 | 0.000520 | 0.448876 | 0.007895 | 98.4% | 3,779 B (15.5x < fp32) |
| WiredCfC + FullyConnected(8, out 1) | 352 | 0.002416 | 0.354496 | 0.029022 | 92.4% | 280 B (5.0x) |
| WiredCfC + NCP(16 inter, 8 cmd, 1 motor) | 2,056 | 0.001180 | 0.514103 | 0.019580 | 96.4% | 964 B (8.5x) |
| LTC + NCP(16 inter, 8 cmd, 1 motor) | 2,781 | 0.000815 | 0.544995 | 0.001454 | 99.9% | 8,623 B |

Measured on a Ryzen 9 7940H (1,600 sequences, 80/20 split, about 4.5 min total). The LTC
cell handles ternarization best: naive PTQ costs it 0.545 MSE, because its
continuous-time dynamics are fragile, and QAT recovers 99.9 % of that gap down to
0.001454. That beats the same-size wired CfC by about 13x, while a 48-step inference runs
in 2.8 ms on the ODE kernels. (The LTC packs to only 1.3x smaller than fp32 because most
of its parameters are the fp32 physical ODE parameters; only the 675 synapse strengths are
ternary.) Every packed artifact round-trips losslessly, and on the exact ncps example
sequence the QAT CfC tracks the double-frequency sine at 0.000603 MSE under deployment
weights.

Deliberate deviations from the reference are documented in `liquid_sine.zig`: the wiring
RNG is the framework's xorshift seeded per wiring (same construction algorithm, different
concrete synapse choices), weights use Xavier-Gauss init, the unwired backbone has 64
units instead of 128, and mixed_memory, the fp32 LSTM augmentation, is not ported.

Run from the repository root:

```bash
zig build --release=fast example-ncps   # writes examples/liquid_sine/modeldata/ncps_{cfc,fc,ncp,ltc}_packed.bin + ncps_valset.bin
make ncps-check                         # C deployment check on all four artifacts
```

The C engine (`c/ncps_inference.c`) recomputes the val MSE of all four artifacts from the
packed trits.
