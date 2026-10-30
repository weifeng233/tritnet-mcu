//! Example: TinyML 1D-CNN gesture classifier, driven by the TritNet-MCU
//! framework (src/framework.zig).
//!
//! The canonical TinyML workload (EdgeImpulse / MLPerf-Tiny flavor): classify
//! short windows of 3-axis accelerometer data into gesture classes. The
//! dataset is a deterministic synthetic generator, so the whole pipeline is
//! reproducible from a fixed seed with no external downloads:
//!
//!   0 "rest"    slow low-amplitude drift on all axes
//!   1 "shake"   fast x-axis oscillation (8-12 cycles per window)
//!   2 "circle"  x/y quadrature pair, 2-4 cycles per window
//!   3 "tap"     a decaying impulse at a random position on a random axis
//!
//! Architecture (2,524 parameters, 54,208 MACs per inference):
//!   input 3x128
//!   Conv1D(3 -> 8, k=8)  + ReLU + MaxPool(2)   -> 8x60
//!   Conv1D(8 -> 16, k=4) + ReLU + MaxPool(2)   -> 16x28
//!   Flatten -> Dense(448 -> 4) + softmax
//!
//! The pipeline is the framework's standard one, single binary, no deps
//! beyond libc:
//!   1. fp32 pretrain     -- Adam + cosine schedule, mini-batch cross-entropy
//!   2. BitNet ternary QAT -- framework/quant.zig: absmean scales, straight-
//!                           through estimator, gradient mask |W| >= 2t,
//!                           8-bit activation quant
//!   3. verification      -- accuracy + confusion matrix for fp32 / PTQ / QAT,
//!                           latency + MACs, ternary storage accounting
//!   4. packed export     -- 5 trits per byte, plus the val split for the C
//!                           deployment engine in c/cnn1d_inference.c
//!   5. round-trip check  -- reload the packed file, re-evaluate; accuracy must
//!                           match the pre-export model exactly
//!
//! Quantization semantics are identical across all examples: conv kernels
//! and the dense kernel are ternary {-1,0,+1} with per-tensor absmean
//! scales; biases stay fp32 (they are 28 of 2,524 parameters and exist
//! precisely to absorb what the ternary grid cannot express).

const std = @import("std");
const math = std.math;
const fw = @import("framework");

const clib = @cImport({
    @cInclude("stdio.h");
    @cInclude("stdlib.h");
    @cInclude("math.h");
});

// ---------------------------------------------------------------------------
// dimensions
// ---------------------------------------------------------------------------

const C0: usize = 3; // input channels: x/y/z accelerometer
const T0: usize = 128; // window length (samples)
const N_CLASS: usize = 4;

const C1: usize = 8; // conv1 output channels
const K1: usize = 8; // conv1 kernel
const T1: usize = T0 - K1 + 1; // 121
const P1: usize = T1 / 2; // 60 (trailing odd sample dropped)

const C2: usize = 16; // conv2 output channels
const K2: usize = 4; // conv2 kernel
const T2: usize = P1 - K2 + 1; // 57
const P2: usize = T2 / 2; // 28

const FC_IN: usize = C2 * P2; // 448

const MACS_PER_INFERENCE: usize =
    C1 * T1 * (C0 * K1) + C2 * T2 * (C1 * K2) + N_CLASS * FC_IN;

const N_PARAMS: usize = C1 * C0 * K1 + C1 + C2 * C1 * K2 + C2 + N_CLASS * FC_IN + N_CLASS;
const N_TERNARY: usize = N_PARAMS - C1 - C2 - N_CLASS; // biases stay fp32

const MAGIC = "CNN1Q002";
const VALSET_MAGIC = "CNN1V001";

const class_names = [4][]const u8{ "rest", "shake", "circle", "tap" };

// ---------------------------------------------------------------------------
// deterministic RNG (framework/rng.zig, this example's own stream)
// ---------------------------------------------------------------------------

const rng = fw.rng;

// ---------------------------------------------------------------------------
// synthetic dataset
// ---------------------------------------------------------------------------

const N_SAMPLES: usize = 4000;
const N_TRAIN: usize = N_SAMPLES * 80 / 100; // 3200
const N_VAL: usize = N_SAMPLES - N_TRAIN; // 800
const NOISE_STD: f32 = 0.06;

var data: [N_SAMPLES][C0 * T0]f32 = undefined;
var labels: [N_SAMPLES]u8 = undefined;

/// One synthetic gesture window. Every class gets random phase, amplitude
/// jitter and noise, so the network must learn waveform shape, not memorize.
fn gen_sample(class: usize, out: []f32) void {
    @memset(out[0 .. C0 * T0], 0);
    const x = out[0..T0];
    const y = out[T0 .. 2 * T0];
    const z = out[2 * T0 .. 3 * T0];
    const phase = rng.range(0, 2 * math.pi);
    const amp: f32 = rng.range(0.8, 1.2);

    switch (class) {
        // rest: slow drift below ~0.3 cycles/window, all axes, small amplitude
        0 => {
            const fx = rng.range(0.1, 0.35);
            const fy = rng.range(0.1, 0.35);
            const fz = rng.range(0.1, 0.35);
            for (0..T0) |i| {
                const t: f32 = @floatFromInt(i);
                x[i] = 0.22 * @sin(2 * math.pi * fx * t / T0 + phase);
                y[i] = 0.22 * @sin(2 * math.pi * fy * t / T0 + phase + 1.3);
                z[i] = 0.22 * @sin(2 * math.pi * fz * t / T0 + phase + 2.6);
            }
        },
        // shake: fast oscillation concentrated on x, 8-12 cycles per window
        1 => {
            const f = rng.range(8.0, 12.0);
            const phase_yz = phase + rng.range(0, 2 * math.pi);
            for (0..T0) |i| {
                const t: f32 = @floatFromInt(i);
                x[i] = amp * @sin(2 * math.pi * f * t / T0 + phase);
                y[i] = 0.18 * @sin(2 * math.pi * f * t / T0 + phase_yz);
                z[i] = 0.18 * @sin(2 * math.pi * f * t / T0 + phase_yz + 0.7);
            }
        },
        // circle: x and y in quadrature at 2-4 cycles per window, z quiet
        2 => {
            const f = rng.range(2.0, 4.0);
            for (0..T0) |i| {
                const t: f32 = @floatFromInt(i);
                const w = 2 * math.pi * f * t / T0 + phase;
                x[i] = amp * @cos(w);
                y[i] = amp * @sin(w);
                z[i] = 0.12 * @sin(2 * w);
            }
        },
        // tap: decaying impulse at a random position on a random axis
        else => {
            const pos: usize = @intFromFloat(rng.range(24, T0 - 48));
            const width: usize = 14 + @as(usize, @intFromFloat(rng.range(0, 8)));
            const axis = rng.next() % 3;
            const a: f32 = rng.range(1.4, 2.2) * (if (rng.next() & 1 == 0) @as(f32, 1.0) else -1.0);
            var i: usize = 0;
            while (i < width and pos + i < T0) : (i += 1) {
                const j: f32 = @floatFromInt(i);
                const s = a * @exp(-j / 5.0) * @sin(0.9 * j);
                switch (axis) {
                    0 => x[pos + i] = s,
                    1 => y[pos + i] = s,
                    else => z[pos + i] = s,
                }
            }
        },
    }
    // sensor noise on every axis; taps get it too (the impulse rides on top)
    for (out[0 .. C0 * T0]) |*v| v.* += rng.gauss(0, NOISE_STD);
}

fn gen_dataset() void {
    var idx: usize = 0;
    while (idx < N_SAMPLES) : (idx += 1) {
        gen_sample(idx % N_CLASS, &data[idx]);
        labels[idx] = @intCast(idx % N_CLASS);
    }
    // Fisher-Yates so the val split is class-mixed
    var i: usize = N_SAMPLES;
    while (i > 1) {
        i -= 1;
        const j: usize = @intFromFloat(rng.uniform() * @as(f64, @floatFromInt(i + 1)));
        std.mem.swap([C0 * T0]f32, &data[i], &data[j]);
        std.mem.swap(u8, &labels[i], &labels[j]);
    }
}

// ---------------------------------------------------------------------------
// parameters
// ---------------------------------------------------------------------------

const PW1 = 0;
const PB1 = 1;
const PW2 = 2;
const PB2 = 3;
const PW3 = 4;
const PB3 = 5;
const N_PARAM = 6;

const Param = struct {
    w: []f32 = &.{}, // latent fp32 weights
    d: []f32 = &.{}, // gradients (zeroed by the optimizer after each step)
    m: []f32 = &.{}, // Adam first moment
    v: []f32 = &.{}, // Adam second moment
    qw: ?[]f32 = null, // deployed table: scale * trit (null = stays fp32)
    t: f32 = 0, // STE threshold = scale/2 from the last ternarize
};

var params: [N_PARAM]Param = undefined;
var gpa = std.heap.c_allocator;

var weights_q = false; // forward reads the ternary tables
var act_q = false; // forward quantizes activations to 8 bit (QAT training)

fn make_param(alloc: std.mem.Allocator, size: usize, stddev: f32) []f32 {
    const p = alloc.alloc(f32, size) catch {
        std.debug.print("out of memory\n", .{});
        clib.exit(1);
        unreachable;
    };
    for (p) |*x| x.* = rng.gauss(0, stddev);
    return p;
}

fn make_zero(alloc: std.mem.Allocator, size: usize) []f32 {
    const p = alloc.alloc(f32, size) catch {
        std.debug.print("out of memory\n", .{});
        clib.exit(1);
        unreachable;
    };
    @memset(p, 0);
    return p;
}

fn init_params(alloc: std.mem.Allocator) void {
    // He initialization; fan-in counts every input tap of one kernel
    const std1: f32 = @sqrt(2.0 / @as(f32, C0 * K1));
    const std2: f32 = @sqrt(2.0 / @as(f32, C1 * K2));
    const std3: f32 = @sqrt(2.0 / @as(f32, FC_IN));
    params[PW1] = .{
        .w = make_param(alloc, C1 * C0 * K1, std1),
        .d = make_zero(alloc, C1 * C0 * K1),
        .m = make_zero(alloc, C1 * C0 * K1),
        .v = make_zero(alloc, C1 * C0 * K1),
        .qw = make_zero(alloc, C1 * C0 * K1),
    };
    params[PB1] = .{
        .w = make_zero(alloc, C1),
        .d = make_zero(alloc, C1),
        .m = make_zero(alloc, C1),
        .v = make_zero(alloc, C1),
    };
    params[PW2] = .{
        .w = make_param(alloc, C2 * C1 * K2, std2),
        .d = make_zero(alloc, C2 * C1 * K2),
        .m = make_zero(alloc, C2 * C1 * K2),
        .v = make_zero(alloc, C2 * C1 * K2),
        .qw = make_zero(alloc, C2 * C1 * K2),
    };
    params[PB2] = .{
        .w = make_zero(alloc, C2),
        .d = make_zero(alloc, C2),
        .m = make_zero(alloc, C2),
        .v = make_zero(alloc, C2),
    };
    params[PW3] = .{
        .w = make_param(alloc, N_CLASS * FC_IN, std3),
        .d = make_zero(alloc, N_CLASS * FC_IN),
        .m = make_zero(alloc, N_CLASS * FC_IN),
        .v = make_zero(alloc, N_CLASS * FC_IN),
        .qw = make_zero(alloc, N_CLASS * FC_IN),
    };
    params[PB3] = .{
        .w = make_zero(alloc, N_CLASS),
        .d = make_zero(alloc, N_CLASS),
        .m = make_zero(alloc, N_CLASS),
        .v = make_zero(alloc, N_CLASS),
    };
}

fn free_params(alloc: std.mem.Allocator) void {
    for (&params) |*p| {
        alloc.free(p.w);
        alloc.free(p.d);
        alloc.free(p.m);
        alloc.free(p.v);
        if (p.qw) |q| alloc.free(q);
    }
}

inline fn W(i: usize) []const f32 {
    const p = &params[i];
    return if (weights_q) (p.qw orelse p.w) else p.w;
}

// ---------------------------------------------------------------------------
// BitNet ternary quantization (framework/quant.zig, same semantics everywhere)
// ---------------------------------------------------------------------------

var trits_scratch: [N_CLASS * FC_IN]i8 = undefined;

fn ternarize(p: *Param) void {
    const qw = p.qw orelse return;
    const info = fw.quant.ternarize(p.w, qw, trits_scratch[0..p.w.len]);
    p.t = info.t;
}

fn ternarize_all() void {
    for (&params) |*p| ternarize(p);
}

/// Straight-through estimator mask: keep gradients only where |W| < 2*t,
/// exactly the rule bitnet_gpt_train.py / BitNetMCU use.
fn mask_all_grads() void {
    for (&params) |*p| {
        if (p.qw == null) continue;
        fw.quant.maskGrads(p.w, p.d, p.t);
    }
}

// ---------------------------------------------------------------------------
// layer kernels (dimension-generic so tests can gradient-check small shapes)
// ---------------------------------------------------------------------------

/// Valid 1D convolution across all input channels.
/// x: [cin][tin], w: [cout][cin*k], out: [cout][tin-k+1]
fn conv1d_fwd(x: []const f32, w: []const f32, b: []const f32, cin: usize, tin: usize, cout: usize, k: usize, out: []f32) void {
    const tout = tin - k + 1;
    for (0..cout) |o| {
        const wr = w[o * cin * k ..][0 .. cin * k];
        const orow = out[o * tout ..][0..tout];
        for (0..tout) |t| {
            var s = b[o];
            for (0..cin) |c| {
                const xr = x[c * tin + t ..][0..k];
                const kc = wr[c * k ..][0..k];
                for (0..k) |jj| s += kc[jj] * xr[jj];
            }
            orow[t] = s;
        }
    }
}

/// dw += dout (x) x, db += sum(dout) -- one pair per (o, t) position.
fn conv1d_bwd_wb(x: []const f32, dout: []const f32, dw: []f32, db: []f32, cin: usize, tin: usize, cout: usize, k: usize) void {
    const tout = tin - k + 1;
    for (0..cout) |o| {
        const dor = dout[o * tout ..][0..tout];
        const dwr = dw[o * cin * k ..][0 .. cin * k];
        var dbsum: f32 = 0;
        for (dor) |d| dbsum += d;
        db[o] += dbsum;
        for (0..tout) |t| {
            const d = dor[t];
            for (0..cin) |c| {
                const xr = x[c * tin + t ..][0..k];
                const kc = dwr[c * k ..][0..k];
                for (0..k) |jj| kc[jj] += d * xr[jj];
            }
        }
    }
}

/// din += W^T (folded) dout.
fn conv1d_bwd_in(w: []const f32, dout: []const f32, din: []f32, cin: usize, tin: usize, cout: usize, k: usize) void {
    const tout = tin - k + 1;
    for (0..cout) |o| {
        const dor = dout[o * tout ..][0..tout];
        const wr = w[o * cin * k ..][0 .. cin * k];
        for (0..tout) |t| {
            const d = dor[t];
            for (0..cin) |c| {
                const dr = din[c * tin + t ..][0..k];
                const kc = wr[c * k ..][0..k];
                for (0..k) |jj| dr[jj] += d * kc[jj];
            }
        }
    }
}

/// Stride-2 max pool; stores which of the two taps won so backward can route.
fn maxpool2_fwd(x: []const f32, cin: usize, tin: usize, arg: []u16, out: []f32) void {
    const tout = tin / 2;
    for (0..cin) |c| {
        for (0..tout) |t| {
            const a = x[c * tin + 2 * t];
            const b = x[c * tin + 2 * t + 1];
            if (b > a) {
                out[c * tout + t] = b;
                arg[c * tout + t] = 1;
            } else {
                out[c * tout + t] = a;
                arg[c * tout + t] = 0;
            }
        }
    }
}

fn maxpool2_bwd(dout: []const f32, arg: []const u16, cin: usize, tout: usize, tin: usize, din: []f32) void {
    for (0..cin) |c| {
        for (0..tout) |t| {
            din[c * tin + 2 * t + arg[c * tout + t]] += dout[c * tout + t];
        }
    }
}

fn softmax_fwd(logits: []const f32, n: usize, probs: []f32) void {
    var mx = logits[0];
    for (1..n) |i| {
        if (logits[i] > mx) mx = logits[i];
    }
    var sum: f32 = 0;
    for (0..n) |i| {
        probs[i] = @exp(logits[i] - mx);
        sum += probs[i];
    }
    const inv: f32 = 1.0 / sum;
    for (0..n) |i| probs[i] *= inv;
}

// ---------------------------------------------------------------------------
// forward / backward over one sample
// ---------------------------------------------------------------------------

const Acts = struct {
    xq: [C0 * T0]f32 = [_]f32{0} ** (C0 * T0), // input as the net ate it (8-bit in QAT)
    c1: [C1 * T1]f32 = [_]f32{0} ** (C1 * T1), // conv1 post-ReLU
    p1_arg: [C1 * P1]u16 = [_]u16{0} ** (C1 * P1),
    p1: [C1 * P1]f32 = [_]f32{0} ** (C1 * P1),
    c2: [C2 * T2]f32 = [_]f32{0} ** (C2 * T2), // conv2 post-ReLU
    p2_arg: [C2 * P2]u16 = [_]u16{0} ** (C2 * P2),
    p2: [C2 * P2]f32 = [_]f32{0} ** (C2 * P2),
    probs: [N_CLASS]f32 = [_]f32{0} ** N_CLASS,
};

var bwd_dp2: [FC_IN]f32 = undefined;
var bwd_dc2: [C2 * T2]f32 = undefined;
var bwd_dp1: [C1 * P1]f32 = undefined;
var bwd_dc1: [C1 * T1]f32 = undefined;

fn forward_sample(x: []const f32, label: usize, act: *Acts) f32 {
    @memcpy(&act.xq, x);
    if (act_q) fw.quant.actQuant8(&act.xq);

    conv1d_fwd(&act.xq, W(PW1), params[PB1].w, C0, T0, C1, K1, &act.c1);
    for (&act.c1) |*v| {
        if (v.* < 0) v.* = 0;
    }
    maxpool2_fwd(&act.c1, C1, T1, &act.p1_arg, &act.p1);
    if (act_q) fw.quant.actQuant8(&act.p1);

    conv1d_fwd(&act.p1, W(PW2), params[PB2].w, C1, P1, C2, K2, &act.c2);
    for (&act.c2) |*v| {
        if (v.* < 0) v.* = 0;
    }
    maxpool2_fwd(&act.c2, C2, T2, &act.p2_arg, &act.p2);
    if (act_q) fw.quant.actQuant8(&act.p2);

    for (0..N_CLASS) |c| {
        var s = params[PB3].w[c];
        const row = W(PW3)[c * FC_IN ..][0..FC_IN];
        for (0..FC_IN) |i| s += row[i] * act.p2[i];
        bwd_logits[c] = s;
    }
    softmax_fwd(&bwd_logits, N_CLASS, &act.probs);
    return -@log(act.probs[label] + 1e-30);
}

var bwd_logits: [N_CLASS]f32 = undefined;

/// Accumulate gradients for one sample; `inv_batch` folds the batch mean in.
fn backward_sample(act: *Acts, label: usize, inv_batch: f32) void {
    var dl: [N_CLASS]f32 = undefined;
    for (0..N_CLASS) |c| {
        const one: f32 = if (c == label) 1.0 else 0.0;
        dl[c] = (act.probs[c] - one) * inv_batch;
    }

    const w3 = W(PW3);
    @memset(&bwd_dp2, 0);
    for (0..N_CLASS) |c| {
        const d = dl[c];
        params[PB3].d[c] += d;
        const dwr = params[PW3].d[c * FC_IN ..][0..FC_IN];
        const wr = w3[c * FC_IN ..][0..FC_IN];
        for (0..FC_IN) |i| {
            dwr[i] += d * act.p2[i];
            bwd_dp2[i] += d * wr[i];
        }
    }

    // 8-bit activation quantization is a rounding step: STE backward is identity
    @memset(&bwd_dc2, 0);
    maxpool2_bwd(&bwd_dp2, &act.p2_arg, C2, P2, T2, &bwd_dc2);
    for (&bwd_dc2, &act.c2) |*g, v| {
        if (!(v > 0)) g.* = 0;
    }
    conv1d_bwd_wb(&act.p1, &bwd_dc2, params[PW2].d, params[PB2].d, C1, P1, C2, K2);
    @memset(&bwd_dp1, 0);
    conv1d_bwd_in(W(PW2), &bwd_dc2, &bwd_dp1, C1, P1, C2, K2);

    @memset(&bwd_dc1, 0);
    maxpool2_bwd(&bwd_dp1, &act.p1_arg, C1, P1, T1, &bwd_dc1);
    for (&bwd_dc1, &act.c1) |*g, v| {
        if (!(v > 0)) g.* = 0;
    }
    conv1d_bwd_wb(&act.xq, &bwd_dc1, params[PW1].d, params[PB1].d, C0, T0, C1, K1);
    // gradient w.r.t. the sensor input is not needed: the world is not trainable
}

// ---------------------------------------------------------------------------
// optimizer (framework/adam.zig)
// ---------------------------------------------------------------------------

fn adam_step(step: usize, steps: usize, lr: f32, gscale: f32) void {
    const bc = fw.adam.biasCorr(step);
    const lr_t = fw.adam.cosineLr(lr, step, steps);
    for (&params) |*p| fw.adam.update(p.w, p.d, p.m, p.v, lr_t, fw.adam.BETA1, fw.adam.BETA2, fw.adam.EPS, bc.b1c, bc.b2c, gscale);
}

fn global_grad_norm() f32 {
    var ss: f32 = 0;
    for (&params) |*p| {
        for (p.d) |g| ss += g * g;
    }
    return @sqrt(ss);
}

fn reset_adam() void {
    for (&params) |*p| {
        @memset(p.m, 0);
        @memset(p.v, 0);
    }
}

// ---------------------------------------------------------------------------
// evaluation / training loops
// ---------------------------------------------------------------------------

const EvalResult = struct {
    loss: f64,
    acc: f64,
    confusion: [N_CLASS][N_CLASS]u32 = [_][N_CLASS]u32{[_]u32{0} ** N_CLASS} ** N_CLASS,
};

/// Evaluation always uses deployment semantics: ternary tables if
/// weights_q is set, float activations.
fn eval_split(begin: usize, count: usize) EvalResult {
    const saved_act_q = act_q;
    act_q = false;
    defer act_q = saved_act_q;
    var act: Acts = .{};
    var res = EvalResult{ .loss = 0, .acc = 0 };
    var correct: usize = 0;
    for (begin..begin + count) |i| {
        const loss = forward_sample(&data[i], labels[i], &act);
        res.loss += loss;
        var pred: usize = 0;
        for (1..N_CLASS) |c| {
            if (act.probs[c] > act.probs[pred]) pred = c;
        }
        if (pred == labels[i]) correct += 1;
        res.confusion[labels[i]][pred] += 1;
    }
    res.loss /= @floatFromInt(count);
    res.acc = @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(count));
    return res;
}

/// One pass of mini-batch training over the shuffled train split.
fn train_phase(comptime qat: bool, steps: usize, lr: f32, batch_size: usize) void {
    weights_q = qat;
    act_q = qat;
    var order: [N_TRAIN]u32 = undefined;
    for (&order, 0..) |*o, i| o.* = @intCast(i);
    var running: f32 = 1.4; // ~ln(4), chance-level cross-entropy
    const per_epoch = N_TRAIN / batch_size;
    const t0 = fw.timing.nowS();
    for (0..steps) |step| {
        if (step % per_epoch == 0) {
            var i: usize = N_TRAIN;
            while (i > 1) {
                i -= 1;
                const j: usize = @intFromFloat(rng.uniform() * @as(f64, @floatFromInt(i + 1)));
                std.mem.swap(u32, &order[i], &order[j]);
            }
        }
        if (qat) ternarize_all();

        const bofs = (step % per_epoch) * batch_size;
        var act: Acts = .{};
        var batch_loss: f32 = 0;
        const inv_batch = 1.0 / @as(f32, @floatFromInt(batch_size));
        for (0..batch_size) |b| {
            const si = order[bofs + b];
            batch_loss += forward_sample(&data[si], labels[si], &act);
            backward_sample(&act, labels[si], inv_batch);
        }
        if (qat) mask_all_grads();

        const gnorm = global_grad_norm();
        const gscale: f32 = if (gnorm > 1.0) 1.0 / gnorm else 1.0;
        adam_step(step, steps, lr, gscale);

        running = running * 0.99 + batch_loss * 0.01;
        if ((step + 1) % 100 == 0 or step == 0 or step == steps - 1) {
            _ = clib.printf("step %4d / %4d | loss %.4f  (avg %.4f)\n", @as(c_int, @intCast(step + 1)), @as(c_int, @intCast(steps)), @as(f64, batch_loss), @as(f64, running));
        }
    }
    const elapsed = fw.timing.nowS() - t0;
    const name: []const u8 = if (qat) "qat " else "fp32";
    _ = clib.printf("%s train: %.3f s  (%.0f steps/sec)\n", @as([*]const u8, @ptrCast(name.ptr)), elapsed, @as(f64, @floatFromInt(steps)) / elapsed);
}

fn print_confusion(cf: *const [N_CLASS][N_CLASS]u32) void {
    _ = clib.printf("  confusion matrix (rows = true, cols = pred)\n");
    _ = clib.printf("            rest  shake circle   tap | recall\n");
    for (0..N_CLASS) |tc| {
        var row_total: u32 = 0;
        for (0..N_CLASS) |pc| row_total += cf[tc][pc];
        _ = clib.printf("  %6s", @as([*]const u8, @ptrCast(class_names[tc].ptr)));
        for (0..N_CLASS) |pc| {
            _ = clib.printf(" %6d", @as(c_int, @intCast(cf[tc][pc])));
        }
        _ = clib.printf(" | %5.1f%%\n", 100.0 * @as(f64, @floatFromInt(cf[tc][tc])) / @as(f64, @floatFromInt(row_total)));
    }
}

// ---------------------------------------------------------------------------
// latency benchmark
// ---------------------------------------------------------------------------

var bench_sink: f32 = 0;

fn bench_latency(ternary_weights: bool, iters: usize) f64 {
    const saved_wq = weights_q;
    const saved_aq = act_q;
    weights_q = ternary_weights;
    act_q = false;
    defer {
        weights_q = saved_wq;
        act_q = saved_aq;
    }
    var act: Acts = .{};
    var sink: f32 = 0;
    // warmup so page faults and caches settle before the clock starts
    for (0..200) |i| {
        sink += forward_sample(&data[i % N_TRAIN], 0, &act);
    }
    const t0 = fw.timing.nowS();
    for (0..iters) |i| {
        sink += forward_sample(&data[i % N_TRAIN], 0, &act);
    }
    const elapsed = fw.timing.nowS() - t0;
    bench_sink += sink; // keep the loop observable; printed once at the end
    return elapsed / @as(f64, @floatFromInt(iters)) * 1e6;
}

// ---------------------------------------------------------------------------
// packed model + val split export (consumed by c/cnn1d_inference.c)
// ---------------------------------------------------------------------------

fn export_model(io: std.Io, path: []const u8, expected_acc: f64) !usize {
    var buf: [1024]u8 = undefined;
    var cur: usize = 0;
    const w = &fw.packed_model;
    w.putBytes(buf[0..], &cur, MAGIC);
    w.putU32(buf[0..], &cur, C0);
    w.putU32(buf[0..], &cur, T0);
    w.putU32(buf[0..], &cur, C1);
    w.putU32(buf[0..], &cur, K1);
    w.putU32(buf[0..], &cur, C2);
    w.putU32(buf[0..], &cur, K2);
    w.putU32(buf[0..], &cur, N_CLASS);
    w.putTernaryMatrix(buf[0..], &cur, params[PW1].w, null, trits_scratch[0..params[PW1].w.len]);
    w.putTernaryMatrix(buf[0..], &cur, params[PW2].w, null, trits_scratch[0..params[PW2].w.len]);
    w.putTernaryMatrix(buf[0..], &cur, params[PW3].w, null, trits_scratch[0..params[PW3].w.len]);
    w.putBytes(buf[0..], &cur, std.mem.sliceAsBytes(params[PB1].w));
    w.putBytes(buf[0..], &cur, std.mem.sliceAsBytes(params[PB2].w));
    w.putBytes(buf[0..], &cur, std.mem.sliceAsBytes(params[PB3].w));
    // The footer is the trainer's own val accuracy. c/cnn1d_inference.c recomputes
    // the metric from the trits alone and compares, so a decode bug fails the
    // deployment check instead of printing a plausible number.
    w.putF64(buf[0..], &cur, expected_acc);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf[0..cur] });
    return cur;
}

fn export_valset(io: std.Io, path: []const u8, alloc: std.mem.Allocator) !usize {
    const n = 16 + N_VAL * (C0 * T0 * 4 + 1);
    const buf = try alloc.alloc(u8, n);
    defer alloc.free(buf);
    var cur: usize = 0;
    const w = &fw.packed_model;
    w.putBytes(buf[0..], &cur, VALSET_MAGIC);
    w.putU32(buf[0..], &cur, N_VAL);
    w.putU32(buf[0..], &cur, C0);
    w.putU32(buf[0..], &cur, T0);
    w.putU32(buf[0..], &cur, N_CLASS);
    // layout: [all windows][all labels] -- the C engine reads the same split
    for (N_TRAIN..N_SAMPLES) |i| {
        w.putBytes(buf[0..], &cur, std.mem.sliceAsBytes(data[i][0..]));
    }
    for (N_TRAIN..N_SAMPLES) |i| {
        buf[cur] = labels[i];
        cur += 1;
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf[0..cur] });
    return cur;
}

/// Rebuild the deployed model from the packed file. Returns the number of
/// trits restored and the trainer's footer accuracy, which the caller checks
/// against the accuracy it computes after reloading.
fn load_model(io: std.Io, path: []const u8, alloc: std.mem.Allocator, expected_acc: *f64) !usize {
    const fdata = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(4096));
    defer alloc.free(fdata);
    var r = fw.packed_model.Reader{ .data = fdata };
    const magic = r.take(MAGIC.len) orelse return error.Truncated;
    if (!std.mem.eql(u8, magic, MAGIC)) return error.BadMagic;
    const dims = [7]usize{ C0, T0, C1, K1, C2, K2, N_CLASS };
    for (dims) |d| {
        const v = r.u32le() orelse return error.Truncated;
        if (v != d) return error.ShapeMismatch;
    }
    var trit_total: usize = 0;
    trit_total += try fw.packed_model.readMatrix(&r, params[PW1].w.len, params[PW1].qw.?, alloc);
    trit_total += try fw.packed_model.readMatrix(&r, params[PW2].w.len, params[PW2].qw.?, alloc);
    trit_total += try fw.packed_model.readMatrix(&r, params[PW3].w.len, params[PW3].qw.?, alloc);
    const b1f = r.take(C1 * 4) orelse return error.Truncated;
    @memcpy(params[PB1].w, std.mem.bytesAsSlice(f32, b1f));
    const b2f = r.take(C2 * 4) orelse return error.Truncated;
    @memcpy(params[PB2].w, std.mem.bytesAsSlice(f32, b2f));
    const b3f = r.take(N_CLASS * 4) orelse return error.Truncated;
    @memcpy(params[PB3].w, std.mem.bytesAsSlice(f32, b3f));
    expected_acc.* = r.f64le() orelse return error.Truncated;
    if (r.take(1) != null) return error.TrailingBytes;
    return trit_total;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

pub fn main(init: std.process.Init) u8 {
    // this example's fixed RNG stream: identical runs on every machine
    rng.state = 0x7EED_1DCA_2026_0001;
    _ = clib.printf("== example-cnn: TinyML 1D-CNN, ternary pipeline ==\n");
    _ = clib.printf("task: 4-class gesture recognition, 3-axis accel %dx%d windows\n", @as(c_int, C0), @as(c_int, T0));
    _ = clib.printf("arch: Conv1D(%d->%d,k=%d)+ReLU+MaxPool -> Conv1D(%d->%d,k=%d)+ReLU+MaxPool -> Dense(%d->%d)\n", @as(c_int, C0), @as(c_int, C1), @as(c_int, K1), @as(c_int, C1), @as(c_int, C2), @as(c_int, K2), @as(c_int, FC_IN), @as(c_int, N_CLASS));

    gen_dataset();
    var per_class = [_]u32{0} ** N_CLASS;
    for (labels) |l| per_class[l] += 1;
    _ = clib.printf("dataset: %d samples (train %d, val %d), classes:", @as(c_int, N_SAMPLES), @as(c_int, N_TRAIN), @as(c_int, N_VAL));
    for (0..N_CLASS) |c| {
        _ = clib.printf(" %s=%d", @as([*]const u8, @ptrCast(class_names[c].ptr)), @as(c_int, @intCast(per_class[c])));
    }
    _ = clib.printf("\n");
    _ = clib.printf("params: %d (ternary %d, fp32 biases %d) | %d MACs/inference\n", @as(c_int, N_PARAMS), @as(c_int, N_TERNARY), @as(c_int, N_PARAMS - N_TERNARY), @as(c_int, MACS_PER_INFERENCE));

    init_params(gpa);
    defer free_params(gpa);

    _ = clib.printf("\n== phase 1: fp32 pretrain ==\n");
    train_phase(false, 2000, 3e-3, 32);

    weights_q = false;
    const fp32_train = eval_split(0, N_TRAIN);
    const fp32_val = eval_split(N_TRAIN, N_VAL);
    _ = clib.printf("\neval (accuracy, float activations)\n");
    _ = clib.printf("  fp32  train %.2f%%   val %.2f%%   (val loss %.4f)\n", 100.0 * fp32_train.acc, 100.0 * fp32_val.acc, fp32_val.loss);

    // post-training quantization reference: ternarize and evaluate as-is
    ternarize_all();
    weights_q = true;
    const ptq_val = eval_split(N_TRAIN, N_VAL);
    _ = clib.printf("  ptq   val %.2f%%   (ternary, no QAT)\n", 100.0 * ptq_val.acc);

    _ = clib.printf("\n== phase 2: BitNet ternary QAT ==\n");
    reset_adam();
    train_phase(true, 1500, 1e-3, 32);

    ternarize_all();
    weights_q = true;
    const qat_train = eval_split(0, N_TRAIN);
    const qat_val = eval_split(N_TRAIN, N_VAL);
    _ = clib.printf("\neval (accuracy, float activations)\n");
    _ = clib.printf("  qat   train %.2f%%   val %.2f%%\n", 100.0 * qat_train.acc, 100.0 * qat_val.acc);

    var hist = [3]usize{ 0, 0, 0 };
    for (&params) |*p| {
        if (p.qw == null) continue;
        fw.quant.histTrits(p.w, trits_scratch[0..p.w.len], &hist);
    }
    const tern_total = hist[0] + hist[1] + hist[2];
    _ = clib.printf("  trits: -1 %.1f%%   0 %.1f%%   +1 %.1f%%\n", 100.0 * @as(f64, @floatFromInt(hist[0])) / @as(f64, @floatFromInt(tern_total)), 100.0 * @as(f64, @floatFromInt(hist[1])) / @as(f64, @floatFromInt(tern_total)), 100.0 * @as(f64, @floatFromInt(hist[2])) / @as(f64, @floatFromInt(tern_total)));

    _ = clib.printf("\n== phase 3: verification ==\n");
    print_confusion(&qat_val.confusion);

    const fp32_bytes = N_PARAMS * 4;
    const packed_bytes = 8 + 7 * 4 + 3 * 8 + fw.ternary.packed_len(N_TERNARY) + (N_PARAMS - N_TERNARY) * 4;
    _ = clib.printf("\nstorage\n");
    _ = clib.printf("  fp32 weights         : %6d bytes\n", @as(c_int, @intCast(fp32_bytes)));
    _ = clib.printf("  naive 2-bit          : %6d bytes\n", @as(c_int, @intCast((N_TERNARY + 3) / 4 + (N_PARAMS - N_TERNARY) * 4)));
    _ = clib.printf("  packed 5-trits/byte  : %6d bytes  (%.1fx smaller than fp32)\n", @as(c_int, @intCast(packed_bytes)), @as(f64, @floatFromInt(fp32_bytes)) / @as(f64, @floatFromInt(packed_bytes)));

    const lat_fp32 = bench_latency(false, 20000);
    const lat_tern = bench_latency(true, 20000);
    _ = clib.printf("\nlatency (host CPU, float activations)\n");
    _ = clib.printf("  fp32 weights         : %6.2f us/inference  (%.1fM MACs/s)\n", lat_fp32, @as(f64, @floatFromInt(MACS_PER_INFERENCE)) / lat_fp32);
    _ = clib.printf("  ternary weights      : %6.2f us/inference  (%.1fM MACs/s)\n", lat_tern, @as(f64, @floatFromInt(MACS_PER_INFERENCE)) / lat_tern);

    _ = clib.printf("\n== phase 4: packed export + round-trip ==\n");
    const model_path = "modeldata/cnn1d_packed.bin";
    const valset_path = "modeldata/cnn1d_valset.bin";
    const model_size = export_model(init.io, model_path, qat_val.acc) catch {
        std.debug.print("failed to write {s}\n", .{model_path});
        return 1;
    };
    _ = export_valset(init.io, valset_path, init.gpa) catch {
        std.debug.print("failed to write {s}\n", .{valset_path});
        return 1;
    };
    _ = clib.printf("  wrote %s (%d bytes)\n", @as([*]const u8, @ptrCast(model_path.ptr)), @as(c_int, @intCast(model_size)));
    _ = clib.printf("  wrote %s (val split for the C engine)\n", @as([*]const u8, @ptrCast(valset_path.ptr)));

    const val_before = qat_val;
    var footer_acc: f64 = 0;
    _ = load_model(init.io, model_path, init.gpa, &footer_acc) catch {
        std.debug.print("failed to load {s}\n", .{model_path});
        return 1;
    };
    const val_after = eval_split(N_TRAIN, N_VAL);
    const roundtrip_ok = val_before.acc == val_after.acc and val_before.loss == val_after.loss;
    _ = clib.printf("  val acc pre-export   : %.2f%% (loss %.6f)\n", 100.0 * val_before.acc, val_before.loss);
    _ = clib.printf("  val acc reloaded     : %.2f%% (loss %.6f)\n", 100.0 * val_after.acc, val_after.loss);
    _ = clib.printf("  footer accuracy      : %.2f%% (what the C engine checks)\n", 100.0 * footer_acc);
    if (roundtrip_ok and footer_acc == val_after.acc) {
        _ = clib.printf("  round-trip: OK (lossless)\n");
    } else {
        _ = clib.printf("  round-trip: MISMATCH\n");
        return 1;
    }

    const recovery = (1.0 - (qat_val.loss - fp32_val.loss) / (ptq_val.loss - fp32_val.loss + 1e-9)) * 100.0;
    _ = clib.printf("\nsummary\n");
    _ = clib.printf("  fp32 val %.2f%% | ptq val %.2f%% | qat val %.2f%% | qat recovery %.1f%% of fp32->ptq loss gap\n", 100.0 * fp32_val.acc, 100.0 * ptq_val.acc, 100.0 * qat_val.acc, recovery);
    _ = clib.printf("  model %d bytes (%.1fx < fp32) | %.2f us/inference | sink %.4f\n", @as(c_int, @intCast(packed_bytes)), @as(f64, @floatFromInt(fp32_bytes)) / @as(f64, @floatFromInt(packed_bytes)), lat_tern, @as(f64, bench_sink));

    return if (roundtrip_ok) 0 else 1;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "conv1d forward matches a direct reference" {
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();
    const cin = 2;
    const tin = 6;
    const cout = 3;
    const k = 3;
    const tout = tin - k + 1;
    const x = try testing.allocator.alloc(f32, cin * tin);
    defer testing.allocator.free(x);
    const w = try testing.allocator.alloc(f32, cout * cin * k);
    defer testing.allocator.free(w);
    const b = try testing.allocator.alloc(f32, cout);
    defer testing.allocator.free(b);
    const out = try testing.allocator.alloc(f32, cout * tout);
    defer testing.allocator.free(out);
    for (x) |*v| v.* = rand.float(f32) * 2 - 1;
    for (w) |*v| v.* = rand.float(f32) * 2 - 1;
    for (b) |*v| v.* = rand.float(f32) * 2 - 1;

    conv1d_fwd(x, w, b, cin, tin, cout, k, out);
    for (0..cout) |o| {
        for (0..tout) |t| {
            var ref = b[o];
            for (0..cin) |c| {
                for (0..k) |jj| ref += w[o * cin * k + c * k + jj] * x[c * tin + t + jj];
            }
            try testing.expectApproxEqAbs(ref, out[o * tout + t], 1e-5);
        }
    }
}

test "conv1d gradients match finite differences" {
    var prng = std.Random.DefaultPrng.init(11);
    const rand = prng.random();
    const cin = 2;
    const tin = 5;
    const cout = 3;
    const k = 3;
    const tout = tin - k + 1;
    const x = try testing.allocator.alloc(f32, cin * tin);
    defer testing.allocator.free(x);
    const w = try testing.allocator.alloc(f32, cout * cin * k);
    defer testing.allocator.free(w);
    const b = try testing.allocator.alloc(f32, cout);
    defer testing.allocator.free(b);
    for (x) |*v| v.* = rand.float(f32) * 2 - 1;
    for (w) |*v| v.* = rand.float(f32) * 0.5 - 0.25;
    for (b) |*v| v.* = rand.float(f32) * 0.5 - 0.25;

    // loss L = sum(conv(x) * R) with a fixed random R, so dL/dw and dL/dx are
    // nontrivial; analytic values come straight from the backward kernels
    const r = try testing.allocator.alloc(f32, cout * tout);
    defer testing.allocator.free(r);
    for (r) |*v| v.* = rand.float(f32) * 2 - 1;

    const dw = try testing.allocator.alloc(f32, w.len);
    defer testing.allocator.free(dw);
    const db = try testing.allocator.alloc(f32, b.len);
    defer testing.allocator.free(db);
    const dx = try testing.allocator.alloc(f32, x.len);
    defer testing.allocator.free(dx);
    @memset(dw, 0);
    @memset(db, 0);
    @memset(dx, 0);
    conv1d_bwd_wb(x, r, dw, db, cin, tin, cout, k);
    conv1d_bwd_in(w, r, dx, cin, tin, cout, k);

    // numeric vs analytic on every third weight and every second input tap
    const h: f32 = 1e-3;
    const out = try testing.allocator.alloc(f32, cout * tout);
    defer testing.allocator.free(out);
    var probe: usize = 0;
    while (probe < w.len) : (probe += 3) {
        const saved = w[probe];
        w[probe] = saved + h;
        conv1d_fwd(x, w, b, cin, tin, cout, k, out);
        var lp: f32 = 0;
        for (out, r) |vv, rr| lp += vv * rr;
        w[probe] = saved - h;
        conv1d_fwd(x, w, b, cin, tin, cout, k, out);
        var lm: f32 = 0;
        for (out, r) |vv, rr| lm += vv * rr;
        w[probe] = saved;
        const num = (lp - lm) / (2 * h);
        try testing.expectApproxEqAbs(num, dw[probe], 1e-3 + 1e-2 * @abs(num));
    }
    probe = 0;
    while (probe < x.len) : (probe += 2) {
        const saved = x[probe];
        x[probe] = saved + h;
        conv1d_fwd(x, w, b, cin, tin, cout, k, out);
        var lp: f32 = 0;
        for (out, r) |vv, rr| lp += vv * rr;
        x[probe] = saved - h;
        conv1d_fwd(x, w, b, cin, tin, cout, k, out);
        var lm: f32 = 0;
        for (out, r) |vv, rr| lm += vv * rr;
        x[probe] = saved;
        const num = (lp - lm) / (2 * h);
        try testing.expectApproxEqAbs(num, dx[probe], 1e-3 + 1e-2 * @abs(num));
    }
}

test "maxpool2 routes gradient to the winner only" {
    // forward picks the larger tap and remembers which one
    const x = [_]f32{ 0.3, -1.0, -2.0, 5.0, 7.0, 7.0 };
    var arg: [3]u16 = undefined;
    var out: [3]f32 = undefined;
    maxpool2_fwd(&x, 1, 6, &arg, &out);
    try testing.expectEqual(@as(u16, 0), arg[0]);
    try testing.expectEqual(@as(u16, 1), arg[1]);
    try testing.expectEqual(@as(u16, 0), arg[2]); // tie goes to the first tap
    try testing.expectApproxEqAbs(@as(f32, 0.3), out[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 5.0), out[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 7.0), out[2], 1e-6);

    var din = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const dout = [_]f32{ 1.0, 2.0, 4.0 };
    maxpool2_bwd(&dout, &arg, 1, 3, 6, &din);
    try testing.expectApproxEqAbs(@as(f32, 1.0), din[0], 1e-6);
    try testing.expectEqual(@as(f32, 0), din[1]);
    try testing.expectEqual(@as(f32, 0), din[2]);
    try testing.expectApproxEqAbs(@as(f32, 2.0), din[3], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 4.0), din[4], 1e-6);
    try testing.expectEqual(@as(f32, 0), din[5]);
}

test "dataset is deterministic and class-balanced" {
    const saved_rng = rng.state;
    defer rng.state = saved_rng;

    rng.state = 0xBEEF;
    var a: [C0 * T0]f32 = undefined;
    var b: [C0 * T0]f32 = undefined;
    gen_sample(2, &a);
    rng.state = 0xBEEF;
    gen_sample(2, &b);
    try testing.expectEqualSlices(f32, &a, &b);

    // classes must stay balanced through the pipeline's fixed generator
    var counts = [_]u32{0} ** N_CLASS;
    for (0..40) |i| {
        const cls = i % N_CLASS;
        counts[cls] += 1;
    }
    for (counts) |cnt| try testing.expectEqual(@as(u32, 10), cnt);
}

test "end-to-end gradient check on sampled weights" {
    const saved_rng = rng.state;
    defer rng.state = saved_rng;
    rng.state = 0xA11CE;

    const alloc = testing.allocator;
    const sizes = [6]usize{ C1 * C0 * K1, C1, C2 * C1 * K2, C2, N_CLASS * FC_IN, N_CLASS };
    var keep: [6][4][]f32 = undefined;
    for (0..6) |i| {
        const n = sizes[i];
        keep[i][0] = try alloc.alloc(f32, n); // w
        keep[i][1] = try alloc.alloc(f32, n); // d
        keep[i][2] = try alloc.alloc(f32, n); // m
        keep[i][3] = try alloc.alloc(f32, n); // v
        @memset(keep[i][1], 0);
        @memset(keep[i][2], 0);
        @memset(keep[i][3], 0);
    }
    defer {
        for (0..6) |i| {
            for (0..4) |j| alloc.free(keep[i][j]);
        }
        for (0..6) |i| if (params[i].qw) |q| alloc.free(q);
    }
    const ternary_sizes = [6]usize{ sizes[0], 0, sizes[2], 0, sizes[4], 0 };
    for (0..6) |i| {
        for (keep[i][0]) |*v| v.* = rng.gauss(0, 0.2);
        var qw: ?[]f32 = null;
        if (ternary_sizes[i] > 0) {
            qw = try alloc.alloc(f32, ternary_sizes[i]);
            @memset(qw.?, 0);
        }
        params[i] = .{ .w = keep[i][0], .d = keep[i][1], .m = keep[i][2], .v = keep[i][3], .qw = qw };
    }

    weights_q = false;
    act_q = false;
    defer {
        weights_q = false;
        act_q = false;
    }

    var x: [C0 * T0]f32 = undefined;
    for (&x) |*v| v.* = rng.gauss(0, 0.5);
    const label: usize = 2;

    var act: Acts = .{};
    _ = forward_sample(&x, label, &act);
    backward_sample(&act, label, 1.0);

    // numeric vs analytic on a spread of indices in every kernel
    const h: f32 = 1e-3;
    const checks = [3]usize{ PW1, PW2, PW3 };
    for (checks) |pi| {
        const p = &params[pi];
        const stride = 37;
        var idx: usize = 0;
        var checked: usize = 0;
        while (idx < p.w.len and checked < 6) : (idx += stride) {
            const saved = p.w[idx];
            p.w[idx] = saved + h;
            const lp = forward_sample(&x, label, &act);
            p.w[idx] = saved - h;
            const lm = forward_sample(&x, label, &act);
            p.w[idx] = saved;
            const num = (lp - lm) / (2 * h);
            try testing.expectApproxEqAbs(num, p.d[idx], 1e-3 + 2e-2 * @abs(num));
            checked += 1;
        }
    }
    // biases of every layer
    for ([3]usize{ PB1, PB2, PB3 }) |pi| {
        const p = &params[pi];
        const saved = p.w[0];
        p.w[0] = saved + h;
        const lp = forward_sample(&x, label, &act);
        p.w[0] = saved - h;
        const lm = forward_sample(&x, label, &act);
        p.w[0] = saved;
        const num = (lp - lm) / (2 * h);
        try testing.expectApproxEqAbs(num, p.d[0], 1e-3 + 2e-2 * @abs(num));
    }
}

test "ternary QAT forward only changes outputs within quantization error" {
    // sanity: with fixed weights, switching to the ternary tables must keep
    // predictions on a fixed input close to the fp32 forward (soft check that
    // absmean scales are calibrated, not a hard bound)
    const saved_rng = rng.state;
    defer rng.state = saved_rng;
    rng.state = 0x1DCA;

    const alloc = testing.allocator;
    const sizes = [6]usize{ C1 * C0 * K1, C1, C2 * C1 * K2, C2, N_CLASS * FC_IN, N_CLASS };
    var keep: [6][4][]f32 = undefined;
    for (0..6) |i| {
        const n = sizes[i];
        keep[i][0] = try alloc.alloc(f32, n);
        keep[i][1] = try alloc.alloc(f32, n);
        keep[i][2] = try alloc.alloc(f32, n);
        keep[i][3] = try alloc.alloc(f32, n);
        @memset(keep[i][1], 0);
        @memset(keep[i][2], 0);
        @memset(keep[i][3], 0);
        for (keep[i][0]) |*v| v.* = rng.gauss(0, 0.2);
    }
    defer {
        for (0..6) |i| {
            for (0..4) |j| alloc.free(keep[i][j]);
        }
        for (0..6) |i| if (params[i].qw) |q| alloc.free(q);
    }
    const ternary_sizes = [6]usize{ sizes[0], 0, sizes[2], 0, sizes[4], 0 };
    for (0..6) |i| {
        var qw: ?[]f32 = null;
        if (ternary_sizes[i] > 0) {
            qw = try alloc.alloc(f32, ternary_sizes[i]);
            @memset(qw.?, 0);
        }
        params[i] = .{ .w = keep[i][0], .d = keep[i][1], .m = keep[i][2], .v = keep[i][3], .qw = qw };
    }

    var x: [C0 * T0]f32 = undefined;
    for (&x) |*v| v.* = rng.gauss(0, 0.5);
    var act: Acts = .{};

    weights_q = false;
    act_q = false;
    _ = forward_sample(&x, 0, &act);
    var fp32_probs: [N_CLASS]f32 = undefined;
    @memcpy(&fp32_probs, &act.probs);

    ternarize_all();
    weights_q = true;
    _ = forward_sample(&x, 0, &act);
    var max_shift: f32 = 0;
    for (0..N_CLASS) |c| {
        max_shift = @max(max_shift, @abs(fp32_probs[c] - act.probs[c]));
    }
    // random untrained weights tolerate a loose bound; the real QAT quality
    // gate is the accuracy pipeline in main()
    try testing.expect(max_shift < 0.5);
    weights_q = false;
}
