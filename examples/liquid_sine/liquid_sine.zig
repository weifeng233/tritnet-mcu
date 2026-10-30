//! Example: Neural Circuit Policies (ncps) on the sine-doubling task,
//! driven by the TritNet-MCU framework (src/framework.zig).
//!
//! A port of the ncps stack (https://github.com/mlech26l/ncps, Lechner &
//! Hasani) onto the framework's ternary MCU pipeline. ncps trains
//! continuous-time RNNs whose sparse "wiring" is a circuit of sensory,
//! inter, command and motor neurons. The two deliverables here are the
//! wiring generator and the CfC cell (Hasani et al., "Closed-form
//! Continuous-time Neural Networks", 2022) -- the closed-form successor of
//! the liquid time-constant (LTC) cell and the cell ncps itself recommends
//! for deployment. Everything runs with nothing beyond libc, one binary:
//!
//!   1. fp32 pretrain     -- Adam + cosine schedule + global-norm clipping,
//!                           hand-written BPTT through every timestep
//!   2. BitNet ternary QAT -- framework/quant.zig: absmean scales, straight-
//!                           through estimator, gradient mask |W| < 2t,
//!                           8-bit activation quant
//!   3. verification      -- val MSE for fp32 / naive PTQ / QAT, ternary
//!                           histogram, storage, latency
//!   4. packed export     -- 5 trits per byte; the NCP sparsity pattern is
//!                           baked into the trits themselves (masked synapses
//!                           have zero latent weights, hence zero trits), so
//!                           the artifact needs no side-car mask
//!   5. round-trip check  -- reload the packed file, re-evaluate; the val MSE
//!                           must match bit-for-bit
//!
//! What is ported from ncps:
//!   - wirings.NCP / AutoNCP  (the 3-layer sensory->inter->command->motor
//!     circuit with stochastic fan-out construction and ±1 polarities)
//!   - wirings.FullyConnected
//!   - torch.CfCCell in all three modes (default / pure / no_gate), with
//!     per-neuron timespans ts, the LeCun-tanh backbone, an optional linear
//!     projection head, and the WiredCfCCell layer stack whose ff1/ff2
//!     projections are masked by the wiring adjacency (ncps leaves the
//!     time-gate projections t_a/t_b dense -- mirrored here)
//!   - the torch_cfc_sinusoidal.py task: a sine/cosine pair in, a
//!     double-frequency sine out, MSE loss -- generalized into a dataset of
//!     random-phase, jittered-frequency sequences so train/val numbers mean
//!     something
//!
//! Deliberate deviations (documented for honesty, not oversight):
//!   - the wiring RNG is the framework's xorshift stream seeded per wiring,
//!     not numpy's RandomState: same construction algorithm, different
//!     concrete synapse choices
//!   - weights use Xavier-Gauss init instead of xavier_uniform_
//!   - mixed_memory (the LSTM augmentation) is not ported; an MCU pipeline
//!     that exists to ternarize projections gains nothing from an fp32 LSTM
//!     riding on top
//!   - the unwired model uses backbone_units=64 rather than ncps' 128 to
//!     keep the host-side training pass quick; the wiring and cell math are
//!     unchanged

const std = @import("std");
const math = std.math;
const fw = @import("framework");

const clib = @cImport({
    @cInclude("stdio.h");
    @cInclude("stdlib.h");
    @cInclude("math.h");
});

const MAGIC = "NCPSQ001";
const VALSET_MAGIC = "NCPSV001";

// ---------------------------------------------------------------------------
// task: ncps torch_cfc_sinusoidal.py, generalized to a dataset
// ---------------------------------------------------------------------------

const T_SEQ: usize = 48; // sequence length (the ncps example's N)
const IN_DIM: usize = 2; // [sin, cos]
const OUT_DIM: usize = 1; // double-frequency sine

const N_SEQ: usize = 1600;
const N_TRAIN: usize = N_SEQ * 80 / 100; // 1280
const N_VAL: usize = N_SEQ - N_TRAIN; // 320
const NOISE_STD: f32 = 0.02;
const FREQ_JITTER: f32 = 0.2; // frequency multiplier k in [1-j, 1+j]

var seq_x: [N_SEQ][T_SEQ * IN_DIM]f32 = undefined;
var seq_y: [N_SEQ][T_SEQ * OUT_DIM]f32 = undefined;

/// One sequence of the sine-doubling family. `k` scales the input frequency,
/// `phase` shifts it; the target is the same waveform at twice the frequency
/// and phase -- exactly the ncps example's relationship (3*pi span in, 6*pi
/// span out). `noise` adds observation noise to the inputs only.
fn gen_sequence(k: f32, phase: f32, noise: f32, x: []f32, y: []f32) void {
    const n1: f32 = @floatFromInt(T_SEQ - 1);
    for (0..T_SEQ) |t| {
        const tf: f32 = @floatFromInt(t);
        const th = 3.0 * math.pi * k * tf / n1 + phase;
        x[2 * t] = @sin(th);
        x[2 * t + 1] = @cos(th);
        y[t] = @sin(2.0 * th);
        if (noise > 0) {
            x[2 * t] += rng.gauss(0, noise);
            x[2 * t + 1] += rng.gauss(0, noise);
        }
    }
}

fn gen_dataset() void {
    for (0..N_SEQ) |i| {
        const k = 1.0 + rng.range(-FREQ_JITTER, FREQ_JITTER);
        const phase = rng.range(0, 2 * math.pi);
        gen_sequence(k, phase, NOISE_STD, &seq_x[i], &seq_y[i]);
    }
    // Fisher-Yates so the val split is i.i.d. with train
    var i: usize = N_SEQ;
    while (i > 1) {
        i -= 1;
        const j: usize = @intFromFloat(rng.uniform() * @as(f64, @floatFromInt(i + 1)));
        std.mem.swap([T_SEQ * IN_DIM]f32, &seq_x[i], &seq_x[j]);
        std.mem.swap([T_SEQ * OUT_DIM]f32, &seq_y[i], &seq_y[j]);
    }
}

// ---------------------------------------------------------------------------
// deterministic RNG (framework/rng.zig, this example's own stream)
// ---------------------------------------------------------------------------

const rng = fw.rng;

// ---------------------------------------------------------------------------
// BitNet ternary quantization (framework/quant.zig, same semantics everywhere)
// ---------------------------------------------------------------------------

const Param = struct {
    w: []f32 = &.{}, // latent fp32 weights
    d: []f32 = &.{}, // gradients (zeroed by the optimizer after each step)
    m: []f32 = &.{}, // Adam first moment
    v: []f32 = &.{}, // Adam second moment
    qw: ?[]f32 = null, // deployed table: scale * trit (null = stays fp32)
    t: f32 = 0, // STE threshold = scale/2 from the last ternarize
    // wiring mask (LTC synapses): masked entries are pinned to zero and the
    // absmean scale is computed over wired entries only, so the zeros never
    // dilute the scale and deployment strength matches fp32
    mask: ?[]const f32 = null,
};

var weights_q = false; // forward reads the ternary tables
var act_q = false; // forward quantizes activations to 8 bit (QAT training)

var trits_scratch: [16384]i8 = undefined;

fn ternarize(p: *Param) void {
    const qw = p.qw orelse return;
    const info = if (p.mask) |mm|
        fw.quant.ternarizeMasked(p.w, mm, qw, trits_scratch[0..p.w.len])
    else
        fw.quant.ternarize(p.w, qw, trits_scratch[0..p.w.len]);
    p.t = info.t;
}

/// Straight-through estimator mask: keep gradients only where |W| < 2*t,
/// exactly the rule the other pipelines (and BitNetMCU) use.
fn mask_grads(p: *Param) void {
    if (p.qw == null) return;
    fw.quant.maskGrads(p.w, p.d, p.t);
}

inline fn lW(p: *const Param) []const f32 {
    return if (weights_q) (p.qw orelse p.w) else p.w;
}

// ---------------------------------------------------------------------------
// wirings (ncps.wirings: NCP / AutoNCP / FullyConnected)
// ---------------------------------------------------------------------------

/// Neuron IDs in the adjacency matrices follow ncps' NCP convention:
/// motor neurons first, then command, then inter. The *state* layout of the
/// wired cell follows the layer processing order: inter, command, motor --
/// so the motor (output) neurons sit at the end of the state vector.
pub const Wiring = struct {
    units: usize,
    input_dim: usize,
    output_dim: usize,
    n_layers: usize,
    adjacency: []i8, // units*units, 0 / ±1 (polarity = erev)
    sensory_adj: []i8, // input_dim*units
    id_start: [3]usize = .{ 0, 0, 0 }, // per layer: first neuron id in adjacency space
    id_len: [3]usize = .{ 0, 0, 0 }, // per layer: neuron count
    slice: [3]Range = .{ .{ .off = 0, .len = 0 }, .{ .off = 0, .len = 0 }, .{ .off = 0, .len = 0 } }, // per layer: state-space range

    pub fn synapseCount(self: *const Wiring) usize {
        var n: usize = 0;
        for (self.adjacency) |a| n += @intCast(@abs(a));
        var m: usize = 0;
        for (self.sensory_adj) |a| m += @intCast(@abs(a));
        return n + m;
    }
};

pub const Range = struct { off: usize, len: usize };

fn alloc_i8(alloc: std.mem.Allocator, n: usize) []i8 {
    const p = alloc.alloc(i8, n) catch {
        std.debug.print("out of memory\n", .{});
        clib.exit(1);
        unreachable;
    };
    @memset(p, 0);
    return p;
}

fn alloc_f32(alloc: std.mem.Allocator, n: usize) []f32 {
    const p = alloc.alloc(f32, n) catch {
        std.debug.print("out of memory\n", .{});
        clib.exit(1);
        unreachable;
    };
    @memset(p, 0);
    return p;
}

/// Chooses `k` distinct entries from `pool` (partial Fisher-Yates).
/// ncps uses np.random.RandomState(seed).choice(..., replace=False); the
/// selection algorithm is mirrored, the concrete choices come from this
/// file's own seeded stream.
fn choice_distinct(pool: []const usize, k: usize, out: []usize) void {
    var scratch: [256]usize = undefined;
    std.debug.assert(pool.len <= scratch.len);
    @memcpy(scratch[0..pool.len], pool);
    for (0..k) |i| {
        const j = i + @as(usize, @intFromFloat(rng.uniform() * @as(f64, @floatFromInt(pool.len - i))));
        std.mem.swap(usize, &scratch[i], &scratch[j]);
        out[i] = scratch[i];
    }
}

inline fn polarity_pm1() i8 {
    return if (rng.next() & 1 == 0) -1 else 1;
}

inline fn polarity_2to1() i8 {
    return if (rng.next() % 3 == 0) -1 else 1; // choice([-1, 1, 1])
}

/// ncps.wirings.NCP: the 3-layer circuit of inter, command and motor neurons
/// with stochastic sensory-fanout / inter-fanout / recurrent-command /
/// motor-fanin construction and reachability repair.
pub fn build_ncp(
    alloc: std.mem.Allocator,
    input_dim: usize,
    inter_neurons: usize,
    command_neurons: usize,
    motor_neurons: usize,
    sensory_fanout: usize,
    inter_fanout: usize,
    recurrent_command_synapses: usize,
    motor_fanin: usize,
    seed: u64,
) Wiring {
    std.debug.assert(motor_fanin <= command_neurons);
    std.debug.assert(sensory_fanout <= inter_neurons);
    std.debug.assert(inter_fanout <= command_neurons);
    const units = inter_neurons + command_neurons + motor_neurons;
    // isolate the wiring's stream from the global one so a wiring depends
    // only on its seed, not on when it is built
    const saved = rng.state;
    rng.state = seed ^ 0xD1B5_9A1D_2026_0001;
    defer rng.state = saved;

    var w = Wiring{
        .units = units,
        .input_dim = input_dim,
        .output_dim = motor_neurons,
        .n_layers = 3,
        .adjacency = alloc_i8(alloc, units * units),
        .sensory_adj = alloc_i8(alloc, input_dim * units),
    };
    const adj = w.adjacency;
    const sens = w.sensory_adj;

    const cmd_start = motor_neurons;
    const inter_start = motor_neurons + command_neurons;

    w.id_start = .{ inter_start, cmd_start, 0 };
    w.id_len = .{ inter_neurons, command_neurons, motor_neurons };
    var off: usize = 0;
    for (0..3) |l| {
        w.slice[l] = .{ .off = off, .len = w.id_len[l] };
        off += w.id_len[l];
    }

    var picked: [256]usize = undefined;
    var inter_pool: [256]usize = undefined;
    var cmd_pool: [256]usize = undefined;
    var sens_pool: [256]usize = undefined;
    var motor_pool: [256]usize = undefined;
    for (0..inter_neurons) |i| inter_pool[i] = inter_start + i;
    for (0..command_neurons) |i| cmd_pool[i] = cmd_start + i;
    for (0..input_dim) |i| sens_pool[i] = i;
    for (0..motor_neurons) |i| motor_pool[i] = i;

    // sensory -> inter: each sensory neuron fans out to sensory_fanout
    // inter neurons; inter neurons that nobody reached get repaired below
    const unreached = alloc_i8(alloc, inter_neurons);
    for (inter_pool[0..inter_neurons]) |id| unreached[id - inter_start] = 1;
    for (0..input_dim) |src| {
        choice_distinct(inter_pool[0..inter_neurons], sensory_fanout, picked[0..sensory_fanout]);
        for (picked[0..sensory_fanout]) |dest| {
            unreached[dest - inter_start] = 0;
            sens[src * units + dest] = polarity_pm1();
        }
    }
    var mean_fanin = input_dim * sensory_fanout / inter_neurons;
    mean_fanin = @max(1, @min(mean_fanin, input_dim));
    for (0..inter_neurons) |d| {
        if (unreached[d] == 0) continue;
        const dest = inter_start + d;
        choice_distinct(sens_pool[0..input_dim], mean_fanin, picked[0..mean_fanin]);
        for (picked[0..mean_fanin]) |src| sens[src * units + dest] = polarity_pm1();
    }

    // inter -> command, same shape
    @memset(unreached[0..command_neurons], 1);
    for (0..inter_neurons) |si| {
        const src = inter_start + si;
        choice_distinct(cmd_pool[0..command_neurons], inter_fanout, picked[0..inter_fanout]);
        for (picked[0..inter_fanout]) |dest| {
            unreached[dest - cmd_start] = 0;
            adj[src * units + dest] = polarity_pm1();
        }
    }
    var mean_cmd_fanin = inter_neurons * inter_fanout / command_neurons;
    mean_cmd_fanin = @max(1, @min(mean_cmd_fanin, inter_neurons));
    for (0..command_neurons) |d| {
        if (unreached[d] == 0) continue;
        const dest = cmd_start + d;
        choice_distinct(inter_pool[0..inter_neurons], mean_cmd_fanin, picked[0..mean_cmd_fanin]);
        for (picked[0..mean_cmd_fanin]) |src| adj[src * units + dest] = polarity_pm1();
    }

    // recurrent command synapses (adjacency entries are set, ncps-style:
    // a repeat pick overwrites, it does not stack)
    for (0..recurrent_command_synapses) |_| {
        const src = cmd_start + @as(usize, @intFromFloat(rng.uniform() * @as(f64, @floatFromInt(command_neurons))));
        const dest = cmd_start + @as(usize, @intFromFloat(rng.uniform() * @as(f64, @floatFromInt(command_neurons))));
        adj[src * units + dest] = polarity_pm1();
    }

    // command -> motor: each motor neuron pulls in motor_fanin command
    // neurons; command neurons that no motor picked get repaired below
    @memset(unreached[0..command_neurons], 1);
    for (0..motor_neurons) |dest| {
        choice_distinct(cmd_pool[0..command_neurons], motor_fanin, picked[0..motor_fanin]);
        for (picked[0..motor_fanin]) |src| {
            unreached[src - cmd_start] = 0;
            adj[src * units + dest] = polarity_pm1();
        }
    }
    var mean_fanout = motor_neurons * motor_fanin / command_neurons;
    mean_fanout = @max(1, @min(mean_fanout, motor_neurons));
    for (0..command_neurons) |s| {
        if (unreached[s] == 0) continue;
        const src = cmd_start + s;
        choice_distinct(motor_pool[0..motor_neurons], mean_fanout, picked[0..mean_fanout]);
        for (picked[0..mean_fanout]) |dest| adj[src * units + dest] = polarity_pm1();
    }

    return w;
}

/// ncps.wirings.AutoNCP: the fanout/fanin parameters derived from just
/// (units, output_size, sparsity_level), exactly as the reference does.
pub fn build_autoncp(alloc: std.mem.Allocator, input_dim: usize, units: usize, output_size: usize, sparsity_level: f32, seed: u64) Wiring {
    std.debug.assert(output_size < units - 2);
    std.debug.assert(sparsity_level >= 0.1 and sparsity_level <= 1.0);
    const density = 1.0 - sparsity_level;
    const inter_and_command = units - output_size;
    const command = @max(inter_and_command * 4 / 10, 1);
    const inter = inter_and_command - command;
    const sensory_fanout = @max(@as(usize, @intFromFloat(@as(f32, @floatFromInt(inter)) * density)), 1);
    const inter_fanout = @max(@as(usize, @intFromFloat(@as(f32, @floatFromInt(command)) * density)), 1);
    const recurrent_cmd = @max(@as(usize, @intFromFloat(2.0 * @as(f32, @floatFromInt(command)) * density)), 1);
    const motor_fanin = @max(@as(usize, @intFromFloat(@as(f32, @floatFromInt(command)) * density)), 1);
    return build_ncp(alloc, input_dim, inter, command, output_size, sensory_fanout, inter_fanout, recurrent_cmd, motor_fanin, seed);
}

/// ncps.wirings.FullyConnected: one layer, every sensory and recurrent pair
/// present, polarity drawn 2:1 in favor of +1.
pub fn build_fully_connected(alloc: std.mem.Allocator, input_dim: usize, units: usize, output_dim: usize, seed: u64) Wiring {
    const saved = rng.state;
    rng.state = seed ^ 0xFCA1_1C0D_2026_0001;
    defer rng.state = saved;
    var w = Wiring{
        .units = units,
        .input_dim = input_dim,
        .output_dim = output_dim,
        .n_layers = 1,
        .adjacency = alloc_i8(alloc, units * units),
        .sensory_adj = alloc_i8(alloc, input_dim * units),
    };
    for (0..units) |src| {
        for (0..units) |dest| w.adjacency[src * units + dest] = polarity_2to1();
    }
    for (0..input_dim) |src| {
        for (0..units) |dest| w.sensory_adj[src * units + dest] = polarity_2to1();
    }
    w.id_start = .{ 0, 0, 0 };
    w.id_len = .{ units, 0, 0 };
    w.slice[0] = .{ .off = 0, .len = units };
    return w;
}

// ---------------------------------------------------------------------------
// CfC cell (ncps.torch.CfCCell) and its linear projections
// ---------------------------------------------------------------------------

pub const Mode = enum(u32) { default = 0, pure = 1, no_gate = 2 };

/// One linear projection: weight [n_out x n_in] plus bias [n_out].
const Lin = struct {
    p: Param,
    b: Param,
    n_in: usize,
    n_out: usize,
};

fn alloc_param(alloc: std.mem.Allocator, n: usize, stddev: f32, comptime kind: enum { gauss, zero, one }, ternary_w: bool) Param {
    const w = alloc_f32(alloc, n);
    switch (kind) {
        .gauss => {
            for (w) |*x| x.* = rng.gauss(0, stddev);
        },
        .zero => @memset(w, 0),
        .one => @memset(w, 1),
    }
    const p = Param{
        .w = w,
        .d = alloc_f32(alloc, n),
        .m = alloc_f32(alloc, n),
        .v = alloc_f32(alloc, n),
        .qw = if (ternary_w) alloc_f32(alloc, n) else null,
    };
    return p;
}

fn alloc_lin(alloc: std.mem.Allocator, n_in: usize, n_out: usize, mask: ?[]const f32) Lin {
    // Xavier-Gauss init for the kernel, 1/sqrt(fan_in) for the bias
    const fan: f32 = @floatFromInt(n_in + n_out);
    const p = alloc_param(alloc, n_out * n_in, @sqrt(2.0 / fan), .gauss, true);
    if (mask) |m| {
        for (p.w, m) |*ww, mm| ww.* *= mm;
    }
    const b = alloc_param(alloc, n_out, 1.0 / @sqrt(@as(f32, @floatFromInt(n_in))), .gauss, false);
    return .{ .p = p, .b = b, .n_in = n_in, .n_out = n_out };
}

/// One CfC cell over a set of neurons. `n_sens` is the number of non-recurrent
/// inputs (sensory inputs, or backbone features for the unwired cell); the
/// recurrent block is always this layer's own previous state, n_out wide.
/// `mask`, when set, restricts the ff1/ff2 projections to wired synapses
/// (ncps applies it there and leaves t_a/t_b dense). Layout is ncps':
/// mask[o][i] with i < n_sens from the (sensory|prev-layer) adjacency and
/// the recurrent block all-ones.
pub const Layer = struct {
    mode: Mode,
    n_sens: usize,
    n_out: usize,
    cat_in: usize, // n_sens + n_out
    ff1: Lin,
    ff2: ?Lin = null,
    t_a: ?Lin = null,
    t_b: ?Lin = null,
    tau: ?Param = null, // pure mode only, fp32 (init 0)
    a_par: ?Param = null, // pure mode only, fp32 (init 1)
    mask: ?[]const f32 = null,
};

fn alloc_layer(alloc: std.mem.Allocator, mode: Mode, n_sens: usize, n_out: usize, mask: ?[]const f32) Layer {
    const cat_in = n_sens + n_out;
    var l = Layer{
        .mode = mode,
        .n_sens = n_sens,
        .n_out = n_out,
        .cat_in = cat_in,
        .ff1 = alloc_lin(alloc, cat_in, n_out, mask),
    };
    if (mode != .pure) {
        l.ff2 = alloc_lin(alloc, cat_in, n_out, mask);
        l.t_a = alloc_lin(alloc, cat_in, n_out, null);
        l.t_b = alloc_lin(alloc, cat_in, n_out, null);
    } else {
        l.tau = alloc_param(alloc, n_out, 0, .zero, false);
        l.a_par = alloc_param(alloc, n_out, 0, .one, false);
    }
    if (mask) |m| l.mask = m;
    return l;
}

/// Per-timestep forward cache for one layer (views into the model's buffers).
const Step = struct {
    xc: []f32, // concat([x_sens, h_prev]) as the projections ate it
    s1: []f32, // tanh(ff1) -- or ff1 raw in pure mode
    s2: []f32, // tanh(ff2) -- or E = exp(-ts*(|tau|+|ff1|)) in pure mode
    gt: []f32, // sigmoid(t_a*ts + t_b)
};

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn lin_fwd(w: []const f32, b: []const f32, x: []const f32, n_in: usize, n_out: usize, out: []f32) void {
    for (0..n_out) |i| {
        var s = b[i];
        const wr = w[i * n_in ..][0..n_in];
        for (wr, x) |ww, xx| s += ww * xx;
        out[i] = s;
    }
}

/// dw += dout (x) x (masked), db += dout, dx += W^T dout (masked).
/// `dx` must be pre-zeroed; pass null when the input gradient is not needed.
fn lin_bwd(w: []const f32, dw: []f32, db: []f32, dout: []const f32, x: []const f32, n_in: usize, n_out: usize, dx: ?[]f32, mask: ?[]const f32) void {
    for (0..n_out) |i| {
        const d = dout[i];
        if (d == 0) continue;
        db[i] += d;
        const wr = w[i * n_in ..][0..n_in];
        const dwr = dw[i * n_in ..][0..n_in];
        if (mask) |m| {
            const mr = m[i * n_in ..][0..n_in];
            for (0..n_in) |j| {
                if (mr[j] == 0) continue;
                dwr[j] += d * x[j];
                if (dx) |dxv| dxv[j] += d * wr[j];
            }
        } else {
            for (0..n_in) |j| {
                dwr[j] += d * x[j];
                if (dx) |dxv| dxv[j] += d * wr[j];
            }
        }
    }
}

/// One cell forward step. h_out = new hidden state (also the layer's output).
fn layer_forward(l: *const Layer, x_sens: []const f32, h_prev: []const f32, ts: f32, st: Step, g_scr: []f32, h_out: []f32) void {
    const n = l.n_out;
    @memcpy(st.xc[0..l.n_sens], x_sens);
    @memcpy(st.xc[l.n_sens..l.cat_in], h_prev);
    if (act_q) fw.quant.actQuant8(st.xc);

    lin_fwd(lW(&l.ff1.p), l.ff1.b.w, st.xc, l.cat_in, n, st.s1);
    switch (l.mode) {
        .pure => {
            // new_hidden = A - A * exp(-ts*(|tau| + |ff1|)) * ff1
            const tau = l.tau.?.w;
            const aa = l.a_par.?.w;
            for (0..n) |i| {
                const e = @exp(-ts * (@abs(tau[i]) + @abs(st.s1[i])));
                st.s2[i] = e;
                h_out[i] = aa[i] - aa[i] * e * st.s1[i];
            }
        },
        .default, .no_gate => {
            for (st.s1) |*v| v.* = math.tanh(v.*);
            const ff2 = &l.ff2.?;
            lin_fwd(lW(&ff2.p), ff2.b.w, st.xc, l.cat_in, n, st.s2);
            for (st.s2) |*v| v.* = math.tanh(v.*);
            const ta = &l.t_a.?;
            const tb = &l.t_b.?;
            lin_fwd(lW(&ta.p), ta.b.w, st.xc, l.cat_in, n, g_scr);
            lin_fwd(lW(&tb.p), tb.b.w, st.xc, l.cat_in, n, st.gt);
            for (0..n) |i| st.gt[i] = sigmoid(g_scr[i] * ts + st.gt[i]);
            if (l.mode == .default) {
                for (0..n) |i| h_out[i] = st.s1[i] * (1.0 - st.gt[i]) + st.gt[i] * st.s2[i];
            } else {
                for (0..n) |i| h_out[i] = st.s1[i] + st.gt[i] * st.s2[i];
            }
        },
    }
}

/// One cell backward step. dh is dL/d(h_out); fills dxcat[0..cat_in] with
/// dL/d(xcat); the caller slices it into dL/d(x_sens) and dL/d(h_prev).
fn layer_backward(l: *const Layer, st: Step, dh: []const f32, ts: f32, sc: [4][]f32, dxcat: []f32) void {
    const n = l.n_out;
    @memset(dxcat[0..l.cat_in], 0);
    const mask = l.mask;
    switch (l.mode) {
        .pure => {
            const tau = &l.tau.?;
            const aa = &l.a_par.?;
            for (0..n) |i| {
                const f = st.s1[i];
                const e = st.s2[i];
                const sf: f32 = if (f > 0) 1 else if (f < 0) @as(f32, -1) else 0;
                const stau: f32 = if (tau.w[i] > 0) 1 else if (tau.w[i] < 0) @as(f32, -1) else 0;
                // dh/df = -A*E*(1 - ts*ff1*sign(ff1))
                sc[0][i] = dh[i] * (-aa.w[i] * e * (1.0 - ts * f * sf));
                tau.d[i] += dh[i] * (aa.w[i] * e * ts * stau);
                aa.d[i] += dh[i] * (1.0 - e * f);
            }
            lin_bwd(lW(&l.ff1.p), l.ff1.p.d, l.ff1.b.d, sc[0][0..n], st.xc, l.cat_in, n, dxcat, mask);
        },
        .default, .no_gate => {
            for (0..n) |i| {
                const a1 = st.s1[i];
                const a2 = st.s2[i];
                const g = st.gt[i];
                const d = dh[i];
                const da1 = if (l.mode == .default) d * (1.0 - g) else d;
                const da2 = d * g;
                const dg = if (l.mode == .default) d * (a2 - a1) else d * a2;
                sc[0][i] = da1 * (1.0 - a1 * a1); // dff1
                sc[1][i] = da2 * (1.0 - a2 * a2); // dff2
                const s_ = dg * g * (1.0 - g);
                sc[2][i] = s_ * ts; // d(t_a)
                sc[3][i] = s_; // d(t_b)
            }
            const ff2 = &l.ff2.?;
            const ta = &l.t_a.?;
            const tb = &l.t_b.?;
            lin_bwd(lW(&l.ff1.p), l.ff1.p.d, l.ff1.b.d, sc[0][0..n], st.xc, l.cat_in, n, dxcat, mask);
            lin_bwd(lW(&ff2.p), ff2.p.d, ff2.b.d, sc[1][0..n], st.xc, l.cat_in, n, dxcat, mask);
            lin_bwd(lW(&ta.p), ta.p.d, ta.b.d, sc[2][0..n], st.xc, l.cat_in, n, dxcat, null);
            lin_bwd(lW(&tb.p), tb.p.d, tb.b.d, sc[3][0..n], st.xc, l.cat_in, n, dxcat, null);
        },
    }
}

// ---------------------------------------------------------------------------
// LTC cell (ncps.torch.LTCCell): the liquid time-constant ODE
// ---------------------------------------------------------------------------
//
// The original Neural Circuit Policies cell: a per-neuron conductance ODE
// solved with `ode_unfolds` Euler fixed-point iterations per timestep:
//
//   w_act[j][i]  = w[j][i] * sigmoid(sigma[j][i]*(v[i]-mu[j][i])) * mask[j][i]
//   numerator    = cm_t*v + gleak*vleak + sum_j w_act*erev (+ sensory term)
//   denominator  = cm_t + gleak + sum_j w_act   (+ sensory term)
//   v           <-  numerator / (denominator + eps)          x ode_unfolds
//
// Matrices are [source x target] over the whole wiring (LTC is inherently
// wired -- there is no unwired LTC). QAT ternarizes the synapse strengths
// w and sensory_w (sparsity again baked into the trits); the physical
// parameters (leak, membrane capacitance, sigmoid center/slope, polarity
// erev, affine I/O mappings) stay fp32 like biases. The backward pass runs
// the chain rule back through all `ode_unfolds` iterations, verified
// against finite differences in the tests.

const LtcCell = struct {
    state_size: usize,
    sensory_size: usize,
    motor_size: usize,
    ode_unfolds: usize = 6,
    epsilon: f32 = 1e-8,
    // ternary synapse strengths
    w: Param, // [state*state], row = source neuron
    sensory_w: Param, // [sens*state]
    // fp32 physical/synapse parameters
    sigma: Param, // [state*state]
    mu: Param,
    erev: Param,
    sensory_sigma: Param, // [sens*state]
    sensory_mu: Param,
    sensory_erev: Param,
    gleak: Param, // [state]
    vleak: Param,
    cm: Param,
    input_w: Param, // [sens]
    input_b: Param,
    output_w: Param, // [motor]
    output_b: Param,
    mask: []const f32, // |adjacency|, [state*state]
    sensory_mask: []const f32,
};

fn alloc_uniform(alloc: std.mem.Allocator, n: usize, lo: f32, hi: f32, ternary_w: bool) Param {
    const w = alloc_f32(alloc, n);
    for (w) |*x| x.* = lo + @as(f32, @floatCast(rng.uniform())) * (hi - lo);
    return .{
        .w = w,
        .d = alloc_f32(alloc, n),
        .m = alloc_f32(alloc, n),
        .v = alloc_f32(alloc, n),
        .qw = if (ternary_w) alloc_f32(alloc, n) else null,
    };
}

/// ncps.torch.LTCCell over a wiring. erev/sensory_erev start at the wiring's
/// ±1 polarities; the rest follows the reference init ranges.
fn alloc_ltc(alloc: std.mem.Allocator, w: *const Wiring) *LtcCell {
    const st = w.units;
    const n_sens = w.input_dim;
    const c = alloc.create(LtcCell) catch unreachable;
    c.* = .{
        .state_size = st,
        .sensory_size = n_sens,
        .motor_size = w.output_dim,
        // masked synapses get zero latents: their gradient is blocked anyway,
        // and zero latents quantize to zero trits, so the packed artifact
        // carries the sparsity pattern with no side-car mask. The absmean
        // scale is computed over wired entries only (framework/quant.zig's
        // masked variant), so
        // the zeros never dilute the scale and deployment strength matches
        // the fp32 pretrain.
        .w = blk: {
            const p = alloc_uniform(alloc, st * st, 0.001, 1.0, true);
            for (p.w, w.adjacency) |*ww, a| {
                if (a == 0) ww.* = 0;
            }
            break :blk p;
        },
        .sensory_w = blk: {
            const p = alloc_uniform(alloc, n_sens * st, 0.001, 1.0, true);
            for (p.w, w.sensory_adj) |*ww, a| {
                if (a == 0) ww.* = 0;
            }
            break :blk p;
        },
        .sigma = alloc_uniform(alloc, st * st, 3.0, 8.0, false),
        .mu = alloc_uniform(alloc, st * st, 0.3, 0.8, false),
        .erev = blk: {
            const p = alloc_uniform(alloc, st * st, 0, 0, false);
            for (p.w, w.adjacency) |*e, a| e.* = @floatFromInt(a);
            break :blk p;
        },
        .sensory_sigma = alloc_uniform(alloc, n_sens * st, 3.0, 8.0, false),
        .sensory_mu = alloc_uniform(alloc, n_sens * st, 0.3, 0.8, false),
        .sensory_erev = blk: {
            const p = alloc_uniform(alloc, n_sens * st, 0, 0, false);
            for (p.w, w.sensory_adj) |*e, a| e.* = @floatFromInt(a);
            break :blk p;
        },
        .gleak = alloc_uniform(alloc, st, 0.001, 1.0, false),
        .vleak = alloc_uniform(alloc, st, -0.2, 0.2, false),
        .cm = alloc_uniform(alloc, st, 0.4, 0.6, false),
        .input_w = blk: {
            const p = alloc_uniform(alloc, n_sens, 0, 0, false);
            @memset(p.w, 1);
            break :blk p;
        },
        .input_b = alloc_uniform(alloc, n_sens, 0, 0, false),
        .output_w = blk: {
            const p = alloc_uniform(alloc, w.output_dim, 0, 0, false);
            @memset(p.w, 1);
            break :blk p;
        },
        .output_b = alloc_uniform(alloc, w.output_dim, 0, 0, false),
        .mask = blk: {
            const m = alloc_f32(alloc, st * st);
            for (m, w.adjacency) |*mm, a| mm.* = @floatFromInt(@abs(a));
            break :blk m;
        },
        .sensory_mask = blk: {
            const m = alloc_f32(alloc, n_sens * st);
            for (m, w.sensory_adj) |*mm, a| mm.* = @floatFromInt(@abs(a));
            break :blk m;
        },
    };
    c.w.mask = c.mask;
    c.sensory_w.mask = c.sensory_mask;
    return c;
}

/// One LTC forward step. `v0` is the incoming state, `v_hist`/`s_u`/`sens_s`
/// are this timestep's cache slices, `xin` receives the mapped input,
/// `v_out` receives the new state. Fills `pred` with the mapped motor output.
fn ltc_forward_step(
    ltc: *const LtcCell,
    x: []const f32,
    v0: []const f32,
    v_hist: []f32, // (unfolds+1) * state
    s_u: []f32, // unfolds * state * state
    sens_s: []f32, // sens * state
    xin: []f32, // sens
    v_out: []f32, // state
    pred: []f32, // motor
) void {
    const st = ltc.state_size;
    const ns = ltc.sensory_size;
    const U = ltc.ode_unfolds;

    // affine input mapping (+ 8-bit activation quantization under QAT)
    for (0..ns) |j| xin[j] = x[j] * ltc.input_w.w[j] + ltc.input_b.w[j];
    if (act_q) fw.quant.actQuant8(xin[0..ns]);
    // synapse strengths follow the deployment tables under QAT (STE): the
    // forward numerics must be exactly what a trit consumer would compute
    const w_dep = lW(&ltc.w);
    const sw_dep = lW(&ltc.sensory_w);

    // sensory synapse activations are loop-invariant: compute once
    var snum: [512]f32 = undefined;
    var sden: [512]f32 = undefined;
    @memset(snum[0..st], 0);
    @memset(sden[0..st], 0);
    for (0..ns) |j| {
        for (0..st) |i| {
            const s = sigmoid(ltc.sensory_sigma.w[j * st + i] * (xin[j] - ltc.sensory_mu.w[j * st + i]));
            sens_s[j * st + i] = s;
            const act = sw_dep[j * st + i] * s * ltc.sensory_mask[j * st + i];
            snum[i] += act * ltc.sensory_erev.w[j * st + i];
            sden[i] += act;
        }
    }

    @memcpy(v_hist[0..st], v0);
    for (0..U) |u| {
        const v_pre = v_hist[u * st ..][0..st];
        const s_row = s_u[u * st * st ..][0 .. st * st];
        var wnum: [512]f32 = undefined;
        var wden: [512]f32 = undefined;
        @memset(wnum[0..st], 0);
        @memset(wden[0..st], 0);
        for (0..st) |j| {
            for (0..st) |i| {
                // gate[j][i] is driven by the SOURCE neuron's voltage v[j]
                // (ncps broadcasts v_pre over the target axis); act[j][i]
                // feeds target i
                s_row[j * st + i] = sigmoid(ltc.sigma.w[j * st + i] * (v_pre[j] - ltc.mu.w[j * st + i]));
                const act = w_dep[j * st + i] * s_row[j * st + i] * ltc.mask[j * st + i];
                wnum[i] += act * ltc.erev.w[j * st + i];
                wden[i] += act;
            }
        }
        for (0..st) |i| {
            const cm_t = ltc.cm.w[i] / (1.0 / @as(f32, @floatFromInt(U)));
            const num = cm_t * v_pre[i] + ltc.gleak.w[i] * ltc.vleak.w[i] + wnum[i] + snum[i];
            const den = cm_t + ltc.gleak.w[i] + wden[i] + sden[i];
            v_hist[(u + 1) * st + i] = num / (den + ltc.epsilon);
        }
    }
    @memcpy(v_out, v_hist[U * st ..][0..st]);
    for (0..ltc.motor_size) |i| {
        pred[i] = v_out[i] * ltc.output_w.w[i] + ltc.output_b.w[i];
    }
}

/// One LTC backward step. `dh_next` holds dL/d(next state from t+1) on entry
/// and is overwritten with dL/d(state at t). Backpropagates the quotient rule
/// through every unfold plus the sensory synapse terms.
fn ltc_backward_step(
    ltc: *const LtcCell,
    x: []const f32, // raw sensory input at this timestep (for d(input_w))
    dpred: []const f32, // motor
    v_hist: []f32,
    s_u: []f32,
    sens_s: []f32,
    xin: []const f32,
    dh_next: []f32, // state; in/out
) void {
    const st = ltc.state_size;
    const ns = ltc.sensory_size;
    const U = ltc.ode_unfolds;

    // output mapping backward: dL/dv_final and the affine parameter grads.
    // Components outside the motor slice carry only the recurrence gradient.
    var dv: [512]f32 = undefined;
    const v_final = v_hist[U * st ..][0..st];
    for (0..st) |i| dv[i] = dh_next[i];
    for (0..ltc.motor_size) |i| {
        dv[i] = dh_next[i] + dpred[i] * ltc.output_w.w[i];
        ltc.output_w.d[i] += dpred[i] * v_final[i];
        ltc.output_b.d[i] += dpred[i];
    }

    // the recompute below must use the same effective weights as the forward
    const w_dep = lW(&ltc.w);
    const sw_dep = lW(&ltc.sensory_w);
    // sensory gradient accumulators (the sensory terms enter every unfold)
    var dsnum: [512]f32 = undefined;
    var dsden: [512]f32 = undefined;
    @memset(dsnum[0..st], 0);
    @memset(dsden[0..st], 0);

    var dnum: [512]f32 = undefined;
    var dden: [512]f32 = undefined;
    var u: usize = U;
    while (u > 0) {
        u -= 1;
        const v_pre = v_hist[u * st ..][0..st];
        const s_row = s_u[u * st * st ..][0 .. st * st];
        // recompute numerator/denominator (cheap) for the quotient rule
        var wnum: [512]f32 = undefined;
        var wden: [512]f32 = undefined;
        var num: [512]f32 = undefined;
        var den: [512]f32 = undefined;
        var snum: [512]f32 = undefined;
        var sden: [512]f32 = undefined;
        @memset(wnum[0..st], 0);
        @memset(wden[0..st], 0);
        @memset(snum[0..st], 0);
        @memset(sden[0..st], 0);
        for (0..st) |j| {
            for (0..st) |i| {
                const act = w_dep[j * st + i] * s_row[j * st + i] * ltc.mask[j * st + i];
                wnum[i] += act * ltc.erev.w[j * st + i];
                wden[i] += act;
            }
        }
        for (0..ns) |j| {
            for (0..st) |i| {
                const act = sw_dep[j * st + i] * sens_s[j * st + i] * ltc.sensory_mask[j * st + i];
                snum[i] += act * ltc.sensory_erev.w[j * st + i];
                sden[i] += act;
            }
        }
        for (0..st) |i| {
            const cm_t = ltc.cm.w[i] / (1.0 / @as(f32, @floatFromInt(U)));
            num[i] = cm_t * v_pre[i] + ltc.gleak.w[i] * ltc.vleak.w[i] + wnum[i] + snum[i];
            den[i] = cm_t + ltc.gleak.w[i] + wden[i] + sden[i];
        }

        for (0..st) |i| {
            const denom = den[i] + ltc.epsilon;
            dnum[i] = dv[i] / denom;
            dden[i] = -dv[i] * num[i] / (denom * denom);
        }
        var dv_new: [512]f32 = undefined;
        @memset(dv_new[0..st], 0);
        for (0..st) |j| {
            for (0..st) |i| {
                const s = s_row[j * st + i];
                // w_act = w * s * mask enters numerator via erev and denominator
                const dw_act = dnum[i] * ltc.erev.w[j * st + i] + dden[i];
                const mm = ltc.mask[j * st + i];
                if (mm != 0) {
                    ltc.w.d[j * st + i] += dw_act * s * mm;
                    ltc.erev.d[j * st + i] += dnum[i] * w_dep[j * st + i] * s * mm;
                }
                const dsig_term = dw_act * w_dep[j * st + i] * mm * s * (1.0 - s);
                ltc.sigma.d[j * st + i] += dsig_term * (v_pre[j] - ltc.mu.w[j * st + i]);
                ltc.mu.d[j * st + i] -= dsig_term * ltc.sigma.w[j * st + i];
                // the gate is driven by the SOURCE voltage: gradient flows to
                // v_pre[j] (ncps broadcasts v_pre over the target axis)
                dv_new[j] += dsig_term * ltc.sigma.w[j * st + i];
            }
        }
        for (0..st) |i| {
            const cm_t = ltc.cm.w[i] / (1.0 / @as(f32, @floatFromInt(U)));
            // d(num)/dv and d(den)/dcm_t -> v and cm
            dv_new[i] += dnum[i] * cm_t;
            const dcm_t = dnum[i] * v_pre[i] + dden[i];
            ltc.cm.d[i] += dcm_t / (1.0 / @as(f32, @floatFromInt(U)));
            ltc.gleak.d[i] += dnum[i] * ltc.vleak.w[i] + dden[i];
            ltc.vleak.d[i] += dnum[i] * ltc.gleak.w[i];
            dsnum[i] += dnum[i];
            dsden[i] += dden[i];
        }
        @memcpy(dv[0..st], dv_new[0..st]);
    }

    // sensory synapse backward (dL/d(input) is not needed: inputs are fixed)
    for (0..ns) |j| {
        for (0..st) |i| {
            const s = sens_s[j * st + i];
            const dw_act = dsnum[i] * ltc.sensory_erev.w[j * st + i] + dsden[i];
            const mm = ltc.sensory_mask[j * st + i];
            if (mm != 0) {
                ltc.sensory_w.d[j * st + i] += dw_act * s * mm;
                ltc.sensory_erev.d[j * st + i] += dsnum[i] * sw_dep[j * st + i] * s * mm;
            }
            const dss = dw_act * sw_dep[j * st + i] * mm * s * (1.0 - s);
            ltc.sensory_sigma.d[j * st + i] += dss * (xin[j] - ltc.sensory_mu.w[j * st + i]);
            ltc.sensory_mu.d[j * st + i] -= dss * ltc.sensory_sigma.w[j * st + i];
        }
    }
    // input mapping backward (affine): only the mapping parameters are trainable
    for (0..ns) |j| {
        var dxin: f32 = 0;
        for (0..st) |i| {
            const s = sens_s[j * st + i];
            const dw_act = dsnum[i] * ltc.sensory_erev.w[j * st + i] + dsden[i];
            dxin += dw_act * sw_dep[j * st + i] * ltc.sensory_mask[j * st + i] * s * (1.0 - s) * ltc.sensory_sigma.w[j * st + i];
        }
        ltc.input_w.d[j] += dxin * x[j];
        ltc.input_b.d[j] += dxin;
    }
    // dL/d(state at t): the ODE's v0 is the previous next_state
    @memcpy(dh_next[0..st], dv[0..st]);
}

// ---------------------------------------------------------------------------
// model: optional backbone -> cell layer stack -> optional projection head
// ---------------------------------------------------------------------------

const LeCunScale: f32 = 1.7159;
const LeCunGain: f32 = 0.666;

const Model = struct {
    name: []const u8,
    mode: Mode,
    in_size: usize,
    out_dim: usize,
    units_total: usize,
    T: usize = T_SEQ,
    bb: ?Lin = null, // backbone over concat([input, h]) (unwired only)
    layers: []Layer,
    proj: ?Lin = null, // linear readout head (unwired only)
    ltc: ?*LtcCell = null, // liquid cell over the whole wiring (LTC models)
    slice: [3]Range = .{ .{ .off = 0, .len = 0 }, .{ .off = 0, .len = 0 }, .{ .off = 0, .len = 0 } },
    out_off: usize = 0, // state offset of the readout (= last layer's slice)
    plist: []*Param = &.{},
    pmask: []?[]const f32 = &.{}, // per plist entry: ff1/ff2 wiring mask if any
    macs_per_seq: usize = 0,

    // caches (T-strided), allocated to the model's shape
    c_xbb: []f32 = &.{}, // T*(in+units): backbone input per step
    c_bbt: []f32 = &.{}, // T*bb: backbone tanh values per step
    dxbb: []f32 = &.{}, // backbone input gradient workspace
    c_xc: [3][]f32 = .{ &.{}, &.{}, &.{} },
    c_s1: [3][]f32 = .{ &.{}, &.{}, &.{} },
    c_s2: [3][]f32 = .{ &.{}, &.{}, &.{} },
    c_gt: [3][]f32 = .{ &.{}, &.{}, &.{} },
    c_h: [3][]f32 = .{ &.{}, &.{}, &.{} },
    c_pred: []f32 = &.{}, // T*out_dim
    // workspaces
    state: []f32 = &.{},
    dstate: []f32 = &.{},
    dxcat: []f32 = &.{},
    bb_out: []f32 = &.{},
    sc: [4][]f32 = .{ &.{}, &.{}, &.{}, &.{} },
    g_scr: []f32 = &.{},
    // LTC caches (T-strided): state through the unfolds, per-unfold sigmoid
    // values, sensory sigmoid values, mapped input
    lt_v: []f32 = &.{}, // T*(U+1)*state
    lt_s: []f32 = &.{}, // T*U*state*state
    lt_ss: []f32 = &.{}, // T*sens*state
    lt_xin: []f32 = &.{}, // T*sens
};

fn collect_params(alloc: std.mem.Allocator, m: *Model) void {
    var n: usize = 0;
    if (m.bb != null) n += 2;
    for (m.layers) |*l| {
        n += 2;
        n += if (l.mode != .pure) @as(usize, 6) else 2;
    }
    if (m.proj != null) n += 2;
    if (m.ltc != null) n += 15;
    const list = alloc.alloc(*Param, n) catch unreachable;
    var i: usize = 0;
    if (m.bb) |*bb| {
        list[i] = &bb.p;
        i += 1;
        list[i] = &bb.b;
        i += 1;
    }
    for (m.layers) |*l| {
        list[i] = &l.ff1.p;
        i += 1;
        list[i] = &l.ff1.b;
        i += 1;
        if (l.mode != .pure) {
            list[i] = &l.ff2.?.p;
            i += 1;
            list[i] = &l.ff2.?.b;
            i += 1;
            list[i] = &l.t_a.?.p;
            i += 1;
            list[i] = &l.t_a.?.b;
            i += 1;
            list[i] = &l.t_b.?.p;
            i += 1;
            list[i] = &l.t_b.?.b;
            i += 1;
        } else {
            list[i] = &l.tau.?;
            i += 1;
            list[i] = &l.a_par.?;
            i += 1;
        }
    }
    if (m.proj) |*pj| {
        list[i] = &pj.p;
        i += 1;
        list[i] = &pj.b;
        i += 1;
    }
    if (m.ltc) |ltc| {
        list[i] = &ltc.w;
        i += 1;
        list[i] = &ltc.sensory_w;
        i += 1;
        list[i] = &ltc.sigma;
        i += 1;
        list[i] = &ltc.mu;
        i += 1;
        list[i] = &ltc.erev;
        i += 1;
        list[i] = &ltc.sensory_sigma;
        i += 1;
        list[i] = &ltc.sensory_mu;
        i += 1;
        list[i] = &ltc.sensory_erev;
        i += 1;
        list[i] = &ltc.gleak;
        i += 1;
        list[i] = &ltc.vleak;
        i += 1;
        list[i] = &ltc.cm;
        i += 1;
        list[i] = &ltc.input_w;
        i += 1;
        list[i] = &ltc.input_b;
        i += 1;
        list[i] = &ltc.output_w;
        i += 1;
        list[i] = &ltc.output_b;
        i += 1;
    }
    m.plist = list;
    const masks = alloc.alloc(?[]const f32, n) catch unreachable;
    i = 0;
    if (m.bb != null) {
        masks[i] = null;
        i += 1;
        masks[i] = null;
        i += 1;
    }
    for (m.layers) |*l| {
        masks[i] = l.mask;
        i += 1;
        masks[i] = null;
        i += 1;
        if (l.mode != .pure) {
            masks[i] = l.mask;
            i += 1;
            masks[i] = null;
            i += 1;
            masks[i] = null;
            i += 1;
            masks[i] = null;
            i += 1;
            masks[i] = null;
            i += 1;
            masks[i] = null;
            i += 1;
        } else {
            masks[i] = null;
            i += 1;
            masks[i] = null;
            i += 1;
        }
    }
    if (m.proj != null) {
        masks[i] = null;
        i += 1;
        masks[i] = null;
        i += 1;
    }
    if (m.ltc) |ltc| {
        const ltc_masks = [15]?[]const f32{
            ltc.mask, ltc.sensory_mask, null, null, null, null, null, null,
            null,     null,             null, null, null, null, null,
        };
        for (ltc_masks) |mm| {
            masks[i] = mm;
            i += 1;
        }
    }
    m.pmask = masks;
}

fn alloc_caches(alloc: std.mem.Allocator, m: *Model) void {
    const T = m.T;
    if (m.bb != null) {
        const bb = &m.bb.?;
        m.c_xbb = alloc_f32(alloc, T * bb.n_in);
        m.c_bbt = alloc_f32(alloc, T * bb.n_out);
        m.bb_out = alloc_f32(alloc, bb.n_out);
        m.dxbb = alloc_f32(alloc, bb.n_in);
    }
    var max_n: usize = 1;
    if (m.bb) |bb| max_n = @max(max_n, bb.n_out);
    for (m.layers, 0..) |*l, li| {
        m.c_xc[li] = alloc_f32(alloc, T * l.cat_in);
        m.c_s1[li] = alloc_f32(alloc, T * l.n_out);
        m.c_s2[li] = alloc_f32(alloc, T * l.n_out);
        m.c_gt[li] = alloc_f32(alloc, T * l.n_out);
        m.c_h[li] = alloc_f32(alloc, T * l.n_out);
        max_n = @max(max_n, l.n_out);
    }
    if (m.ltc) |ltc| {
        const st = ltc.state_size;
        const ns = ltc.sensory_size;
        const U = ltc.ode_unfolds;
        m.lt_v = alloc_f32(alloc, T * (U + 1) * st);
        m.lt_s = alloc_f32(alloc, T * U * st * st);
        m.lt_ss = alloc_f32(alloc, T * ns * st);
        m.lt_xin = alloc_f32(alloc, T * ns);
        max_n = @max(max_n, m.out_dim);
    }
    m.c_pred = alloc_f32(alloc, T * m.out_dim);
    m.state = alloc_f32(alloc, m.units_total);
    m.dstate = alloc_f32(alloc, m.units_total);
    var max_cat: usize = m.in_size;
    for (m.layers) |*l| max_cat = @max(max_cat, l.cat_in);
    m.dxcat = alloc_f32(alloc, max_cat);
    for (0..4) |i| m.sc[i] = alloc_f32(alloc, max_n);
    m.g_scr = alloc_f32(alloc, max_n);

    // MAC accounting: one dot product of length n_in per output per projection
    var macs: usize = 0;
    if (m.bb) |bb| macs += bb.n_out * bb.n_in;
    for (m.layers) |*l| {
        const per_proj: usize = if (l.mode == .pure) 1 else 4;
        macs += per_proj * l.n_out * l.cat_in;
    }
    if (m.proj) |pj| macs += pj.n_out * pj.n_in;
    if (m.ltc) |ltc| {
        // per unfold: two reductions over the synapse matrix + sensory terms
        const U = ltc.ode_unfolds;
        const st = ltc.state_size;
        const ns = ltc.sensory_size;
        macs += U * 2 * st * (st + ns) + ns * st + ltc.motor_size;
    }
    m.macs_per_seq = m.T * macs;
}

fn step_of(m: *Model, li: usize, t: usize) Step {
    const l = &m.layers[li];
    const n = l.n_out;
    return .{
        .xc = m.c_xc[li][t * l.cat_in ..][0..l.cat_in],
        .s1 = m.c_s1[li][t * n ..][0..n],
        .s2 = m.c_s2[li][t * n ..][0..n],
        .gt = m.c_gt[li][t * n ..][0..n],
    };
}

/// Full-sequence forward. Fills m.c_pred with one row per timestep.
fn forward_seq(m: *Model, x: []const f32) void {
    const T = m.T;
    @memset(m.state, 0);
    for (0..T) |t| {
        var layer_in: []const f32 = x[t * m.in_size ..][0..m.in_size];
        if (m.ltc != null) {
            // LTC is one cell over the whole wiring: the state is the ODE
            // state and the readout is the motor slice (state[0..motor])
            const ltc = m.ltc.?;
            const st = ltc.state_size;
            const ns = ltc.sensory_size;
            const U = ltc.ode_unfolds;
            const v_hist = m.lt_v[t * (U + 1) * st ..][0 .. (U + 1) * st];
            const s_u = m.lt_s[t * U * st * st ..][0 .. U * st * st];
            const sens_s = m.lt_ss[t * ns * st ..][0 .. ns * st];
            const xin = m.lt_xin[t * ns ..][0..ns];
            const pred = m.c_pred[t * m.out_dim ..][0..m.out_dim];
            ltc_forward_step(ltc, layer_in, m.state, v_hist, s_u, sens_s, xin, m.state, pred);
            continue;
        }
        if (m.bb) |*bb| {
            const xbb = m.c_xbb[t * bb.n_in ..][0..bb.n_in];
            @memcpy(xbb[0..m.in_size], layer_in);
            @memcpy(xbb[m.in_size..], m.state);
            if (act_q) fw.quant.actQuant8(xbb);
            lin_fwd(lW(&bb.p), bb.b.w, xbb, bb.n_in, bb.n_out, m.bb_out);
            for (m.bb_out, 0..) |*v, i| {
                const tv = math.tanh(LeCunGain * v.*);
                m.c_bbt[t * bb.n_out + i] = tv;
                v.* = LeCunScale * tv;
            }
            layer_in = m.bb_out;
        }
        for (m.layers, 0..) |*l, li| {
            const st = step_of(m, li, t);
            const h_prev = m.state[m.slice[li].off..][0..l.n_out];
            const hout = m.c_h[li][t * l.n_out ..][0..l.n_out];
            layer_forward(l, layer_in, h_prev, 1.0, st, m.g_scr, hout);
            layer_in = hout;
        }
        // publish this timestep's layer outputs into the state vector
        for (m.layers, 0..) |*l, li| {
            @memcpy(m.state[m.slice[li].off..][0..l.n_out], m.c_h[li][t * l.n_out ..][0..l.n_out]);
        }
        if (m.proj) |*pj| {
            const hp = m.c_h[0][t * m.units_total ..][0..m.units_total];
            lin_fwd(lW(&pj.p), pj.b.w, hp, pj.n_in, pj.n_out, m.c_pred[t * m.out_dim ..][0..m.out_dim]);
        } else {
            @memcpy(m.c_pred[t * m.out_dim ..][0..m.out_dim], m.state[m.out_off..][0..m.out_dim]);
        }
    }
}

/// Full-sequence backward (BPTT). dy holds the per-timestep targets;
/// inv_elems scales the MSE gradient (batch mean / (T*out_dim) folded in
/// by the caller). Accumulates into every param's `.d`.
fn backward_seq(m: *Model, x: []const f32, y: []const f32, inv_elems: f32) void {
    const T = m.T;
    const out = m.out_dim;
    @memset(m.dstate, 0);
    var t: usize = T;
    while (t > 0) {
        t -= 1;
        const pred = m.c_pred[t * out ..][0..out];
        const yy = y[t * out ..][0..out];

        if (m.ltc) |ltc| {
            // dpred at t + the recurrence gradient carried in m.dstate;
            // ltc_backward_step overwrites m.dstate with dL/d(state at t-1)
            const st = ltc.state_size;
            const ns = ltc.sensory_size;
            const U = ltc.ode_unfolds;
            const v_hist = m.lt_v[t * (U + 1) * st ..][0 .. (U + 1) * st];
            const s_u = m.lt_s[t * U * st * st ..][0 .. U * st * st];
            const sens_s = m.lt_ss[t * ns * st ..][0 .. ns * st];
            const xin = m.lt_xin[t * ns ..][0..ns];
            for (0..out) |i| m.sc[0][i] = 2.0 * (pred[i] - yy[i]) * inv_elems;
            ltc_backward_step(ltc, x[t * m.in_size ..][0..m.in_size], m.sc[0][0..out], v_hist, s_u, sens_s, xin, m.dstate);
            continue;
        }
        if (m.proj) |*pj| {
            // head: dpred -> dproj_w, dh = projW^T dpred (into m.sc[1])
            for (0..out) |i| m.sc[0][i] = 2.0 * (pred[i] - yy[i]) * inv_elems;
            const h = m.c_h[0][t * m.units_total ..][0..m.units_total];
            @memset(m.sc[1][0..m.units_total], 0);
            lin_bwd(lW(&pj.p), pj.p.d, pj.b.d, m.sc[0][0..out], h, m.units_total, out, m.sc[1][0..m.units_total], null);
            for (0..m.units_total) |i| m.dstate[i] += m.sc[1][i];
        } else {
            for (0..out) |i| m.dstate[m.out_off + i] += 2.0 * (pred[i] - yy[i]) * inv_elems;
        }

        // layers, same-timestep gradient flows last -> first
        var li: usize = m.layers.len;
        while (li > 0) {
            li -= 1;
            const l = &m.layers[li];
            const sl = m.slice[li];
            layer_backward(l, step_of(m, li, t), m.dstate[sl.off..][0..l.n_out], 1.0, m.sc, m.dxcat);
            // dstate[sl] for t-1 is exactly this layer's d(h_prev)
            @memcpy(m.dstate[sl.off..][0..l.n_out], m.dxcat[l.n_sens..][0..l.n_out]);
            if (li > 0) {
                const prev = &m.layers[li - 1];
                for (0..prev.n_out) |i| m.dstate[m.slice[li - 1].off + i] += m.dxcat[i];
            }
        }

        // backbone (unwired only): d(bb_out) -> LeCun' -> dkernel; the input
        // gradient's h-part flows back into the previous state (the backbone
        // consumes concat([input, h]), so h_{t-1} carries gradient through it)
        if (m.bb) |*bb| {
            const dlecun = LeCunScale * LeCunGain;
            const n = bb.n_out;
            for (0..n) |i| {
                const tv = m.c_bbt[t * n + i];
                m.sc[0][i] = m.dxcat[i] * dlecun * (1.0 - tv * tv);
            }
            const xbb = m.c_xbb[t * bb.n_in ..][0..bb.n_in];
            @memset(m.dxbb, 0);
            lin_bwd(lW(&bb.p), bb.p.d, bb.b.d, m.sc[0][0..n], xbb, bb.n_in, n, m.dxbb, null);
            for (0..m.units_total) |i| m.dstate[i] += m.dxbb[m.in_size + i];
        }
    }
}

// ---------------------------------------------------------------------------
// model builders
// ---------------------------------------------------------------------------

/// ncps.torch.CfC: backbone(input+h -> bb_units, LeCun tanh) -> CfCCell ->
/// optional linear projection head.
fn build_unwired(alloc: std.mem.Allocator, name: []const u8, in_size: usize, units: usize, bb_units: usize, proj_size: usize, mode: Mode) *Model {
    const m = alloc.create(Model) catch unreachable;
    m.* = .{
        .name = name,
        .mode = mode,
        .in_size = in_size,
        .out_dim = if (proj_size > 0) proj_size else units,
        .units_total = units,
        .layers = &.{},
    };
    const layers = alloc.alloc(Layer, 1) catch unreachable;
    // no backbone: the cell consumes [input, h] directly
    const n_sens = if (bb_units > 0) bb_units else in_size;
    layers[0] = alloc_layer(alloc, mode, n_sens, units, null);
    m.layers = layers;
    m.slice[0] = .{ .off = 0, .len = units };
    if (bb_units > 0) m.bb = alloc_lin(alloc, in_size + units, bb_units, null);
    if (proj_size > 0 and proj_size != units) m.proj = alloc_lin(alloc, units, proj_size, null);
    m.out_off = 0;
    collect_params(alloc, m);
    alloc_caches(alloc, m);
    return m;
}

/// ncps.torch.WiredCfC: one CfCCell per wiring layer, ff1/ff2 masked by the
/// wiring's adjacency (sensory adjacency for layer 0), recurrent block dense.
fn build_wired(alloc: std.mem.Allocator, name: []const u8, w: *const Wiring, mode: Mode) *Model {
    const m = alloc.create(Model) catch unreachable;
    m.* = .{
        .name = name,
        .mode = mode,
        .in_size = w.input_dim,
        .out_dim = w.output_dim,
        .units_total = w.units,
        .layers = &.{},
    };
    const layers = alloc.alloc(Layer, w.n_layers) catch unreachable;
    for (0..w.n_layers) |l| {
        const n_out = w.slice[l].len;
        const n_prev = if (l == 0) w.input_dim else w.slice[l - 1].len;
        const cat_in = n_prev + n_out;
        const mask = alloc_f32(alloc, n_out * cat_in);
        for (0..n_out) |o| {
            const dest_id = w.id_start[l] + o;
            for (0..n_prev) |i| {
                const a: i8 = if (l == 0)
                    w.sensory_adj[i * w.units + dest_id]
                else
                    w.adjacency[(w.id_start[l - 1] + i) * w.units + dest_id];
                mask[o * cat_in + i] = @floatFromInt(@abs(a));
            }
            for (n_prev..cat_in) |i| mask[o * cat_in + i] = 1.0;
        }
        layers[l] = alloc_layer(alloc, mode, n_prev, n_out, mask);
    }
    m.layers = layers;
    m.slice = w.slice;
    m.out_off = w.slice[w.n_layers - 1].off;
    collect_params(alloc, m);
    alloc_caches(alloc, m);
    return m;
}

/// ncps.torch.LTC: a single LTCCell over the whole wiring. The state vector
/// follows the wiring's neuron-ID order (motor first), and the readout is the
/// motor slice state[0..motor], matching LTCCell._map_outputs.
fn build_ltc(alloc: std.mem.Allocator, name: []const u8, w: *const Wiring) *Model {
    const m = alloc.create(Model) catch unreachable;
    m.* = .{
        .name = name,
        .mode = .default,
        .in_size = w.input_dim,
        .out_dim = w.output_dim,
        .units_total = w.units,
        .layers = &.{},
        .ltc = alloc_ltc(alloc, w),
    };
    m.slice[0] = .{ .off = 0, .len = w.units };
    m.out_off = 0;
    collect_params(alloc, m);
    alloc_caches(alloc, m);
    return m;
}

// ---------------------------------------------------------------------------
// ternary helpers over a whole model
// ---------------------------------------------------------------------------

fn ternarize_model(m: *Model) void {
    for (m.plist) |p| ternarize(p);
}

fn mask_model_grads(m: *Model) void {
    for (m.plist) |p| mask_grads(p);
}

fn zero_model_grads(m: *Model) void {
    for (m.plist) |p| @memset(p.d, 0);
}

fn global_grad_norm(m: *Model) f32 {
    var ss: f32 = 0;
    for (m.plist) |p| {
        for (p.d) |g| ss += g * g;
    }
    return @sqrt(ss);
}

// ---------------------------------------------------------------------------
// optimizer (framework/adam.zig: Adam + cosine schedule + global-norm clip)
// ---------------------------------------------------------------------------

fn adam_step(m: *Model, step: usize, steps: usize, lr: f32, gscale: f32) void {
    const bc = fw.adam.biasCorr(step);
    const lr_t = fw.adam.cosineLr(lr, step, steps);
    for (m.plist) |p| fw.adam.update(p.w, p.d, p.m, p.v, lr_t, fw.adam.BETA1, fw.adam.BETA2, fw.adam.EPS, bc.b1c, bc.b2c, gscale);
}

fn reset_adam(m: *Model) void {
    for (m.plist) |p| {
        @memset(p.m, 0);
        @memset(p.v, 0);
    }
}

// ---------------------------------------------------------------------------
// training / evaluation
// ---------------------------------------------------------------------------

/// Mean MSE over a slice of the dataset under the current weight mode.
/// Deployment evaluation always runs with float activations.
fn eval_mse(m: *Model, begin: usize, count: usize) f64 {
    const saved = act_q;
    act_q = false;
    defer act_q = saved;
    var ss: f64 = 0;
    for (begin..begin + count) |s| {
        forward_seq(m, &seq_x[s]);
        const y = &seq_y[s];
        for (0..m.T * m.out_dim) |i| {
            const d = m.c_pred[i] - y[i];
            ss += d * d;
        }
    }
    return ss / @as(f64, @floatFromInt(count * m.T * m.out_dim));
}

/// One training phase: shuffled mini-batches, per-sample BPTT accumulation,
/// optional ternary QAT (tables refreshed per step, STE grad mask), global
/// norm clipping to 1 (ncps' gradient_clip_val) and Adam with cosine decay.
fn train_phase(m: *Model, comptime qat: bool, steps: usize, lr: f32, batch: usize) void {
    weights_q = qat;
    act_q = qat;
    var order: [N_TRAIN]u32 = undefined;
    for (&order, 0..) |*o, i| o.* = @intCast(i);
    const inv_elems = 1.0 / @as(f32, @floatFromInt(batch * m.T * m.out_dim));
    const per_epoch = N_TRAIN / batch;
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
        if (qat) ternarize_model(m);

        const bofs = (step % per_epoch) * batch;
        for (0..batch) |b| {
            const si = order[bofs + b];
            forward_seq(m, &seq_x[si]);
            backward_seq(m, &seq_x[si], &seq_y[si], inv_elems);
        }
        if (qat) mask_model_grads(m);

        const gnorm = global_grad_norm(m);
        const gscale: f32 = if (gnorm > 1.0) 1.0 / gnorm else 1.0;
        adam_step(m, step, steps, lr, gscale);

        if ((step + 1) % 200 == 0 or step == 0 or step == steps - 1) {
            // refresh running loss on a fixed probe batch (deployment tables)
            const saved_wq = weights_q;
            const saved_aq = act_q;
            weights_q = qat;
            act_q = false;
            forward_seq(m, &seq_x[order[0]]);
            var bl: f64 = 0;
            const y = &seq_y[order[0]];
            for (0..m.T * m.out_dim) |i| {
                const d = m.c_pred[i] - y[i];
                bl += d * d;
            }
            weights_q = saved_wq;
            act_q = saved_aq;
            _ = clib.printf("  step %4d / %4d | mse(sample) %.6f\n", @as(c_int, @intCast(step + 1)), @as(c_int, @intCast(steps)), bl / @as(f64, @floatFromInt(m.T * m.out_dim)));
        }
    }
    const elapsed = fw.timing.nowS() - t0;
    const name: []const u8 = if (qat) "qat " else "fp32";
    _ = clib.printf("  %s train: %.2f s  (%.0f steps/sec)\n", @as([*]const u8, @ptrCast(name.ptr)), elapsed, @as(f64, @floatFromInt(steps)) / elapsed);
}

// ---------------------------------------------------------------------------
// packed model + val split export (framework/packed_model.zig; consumed by
// c/ncps_inference.c)
// ---------------------------------------------------------------------------

/// One ternary matrix record. The wiring mask, when present, selects the
/// wired-only absmean variant: the sparsity pattern rides in the trits.
fn put_ternary_matrix(buf: []u8, cur: *usize, p: *Param) void {
    fw.packed_model.putTernaryMatrix(buf[0..], cur, p.w, p.mask, trits_scratch[0..p.w.len]);
}

fn put_bias(buf: []u8, cur: *usize, p: *Param) void {
    fw.packed_model.putBytes(buf, cur, std.mem.sliceAsBytes(p.w));
}

fn put_lin(buf: []u8, cur: *usize, l: *Lin) void {
    put_ternary_matrix(buf, cur, &l.p);
    put_bias(buf, cur, &l.b);
}

fn export_model(io: std.Io, alloc: std.mem.Allocator, m: *Model, path: []const u8, expected_mse: f64) !usize {
    const buf = try alloc.alloc(u8, 1 << 20);
    defer alloc.free(buf);
    var cur: usize = 0;
    fw.packed_model.putBytes(buf[0..], &cur, MAGIC);
    fw.packed_model.putU32(buf[0..], &cur, @intCast(m.in_size));
    fw.packed_model.putU32(buf[0..], &cur, @intCast(m.units_total));
    fw.packed_model.putU32(buf[0..], &cur, @intCast(m.out_dim));
    fw.packed_model.putU32(buf[0..], &cur, @intCast(m.layers.len));
    fw.packed_model.putU32(buf[0..], &cur, @intFromEnum(m.mode));
    fw.packed_model.putU32(buf[0..], &cur, if (m.bb != null) @intCast(m.bb.?.n_out) else 0);
    fw.packed_model.putU32(buf[0..], &cur, if (m.proj != null) @intCast(m.proj.?.n_out) else 0);
    fw.packed_model.putU32(buf[0..], &cur, if (m.ltc != null) 1 else 0); // cell kind
    if (m.ltc) |ltc| {
        // liquid cell: ternary synapse strengths + fp32 physical parameters
        put_ternary_matrix(buf[0..], &cur, &ltc.w);
        put_ternary_matrix(buf[0..], &cur, &ltc.sensory_w);
        const fps = [13]*Param{ &ltc.sigma, &ltc.mu, &ltc.erev, &ltc.sensory_sigma, &ltc.sensory_mu, &ltc.sensory_erev, &ltc.gleak, &ltc.vleak, &ltc.cm, &ltc.input_w, &ltc.input_b, &ltc.output_w, &ltc.output_b };
        for (fps) |p| fw.packed_model.putBytes(buf[0..], &cur, std.mem.sliceAsBytes(p.w));
    }
    if (m.bb) |*bb| put_lin(buf[0..], &cur, bb);
    for (m.layers) |*l| {
        fw.packed_model.putU32(buf[0..], &cur, @intCast(l.n_sens));
        fw.packed_model.putU32(buf[0..], &cur, @intCast(l.n_out));
        put_lin(buf[0..], &cur, &l.ff1);
        if (l.mode != .pure) {
            put_lin(buf[0..], &cur, &l.ff2.?);
            put_lin(buf[0..], &cur, &l.t_a.?);
            put_lin(buf[0..], &cur, &l.t_b.?);
        } else {
            fw.packed_model.putBytes(buf[0..], &cur, std.mem.sliceAsBytes(l.tau.?.w));
            fw.packed_model.putBytes(buf[0..], &cur, std.mem.sliceAsBytes(l.a_par.?.w));
        }
    }
    if (m.proj) |*pj| put_lin(buf[0..], &cur, pj);
    fw.packed_model.putF64(buf[0..], &cur, expected_mse);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf[0..cur] });
    return cur;
}

fn export_valset(io: std.Io, alloc: std.mem.Allocator, path: []const u8) !usize {
    const n = 24 + N_VAL * (T_SEQ * IN_DIM + T_SEQ * OUT_DIM) * 4;
    const buf = try alloc.alloc(u8, n);
    defer alloc.free(buf);
    var cur: usize = 0;
    fw.packed_model.putBytes(buf[0..], &cur, VALSET_MAGIC);
    fw.packed_model.putU32(buf[0..], &cur, @intCast(N_VAL));
    fw.packed_model.putU32(buf[0..], &cur, @intCast(T_SEQ));
    fw.packed_model.putU32(buf[0..], &cur, @intCast(IN_DIM));
    fw.packed_model.putU32(buf[0..], &cur, @intCast(OUT_DIM));
    for (N_TRAIN..N_SEQ) |i| {
        fw.packed_model.putBytes(buf[0..], &cur, std.mem.sliceAsBytes(seq_x[i][0..]));
    }
    for (N_TRAIN..N_SEQ) |i| {
        fw.packed_model.putBytes(buf[0..], &cur, std.mem.sliceAsBytes(seq_y[i][0..]));
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf[0..cur] });
    return cur;
}

/// Rebuild the deployed model from the packed file (round-trip check).
fn load_model(io: std.Io, alloc: std.mem.Allocator, m: *Model, path: []const u8) !void {
    const fdata = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 22));
    defer alloc.free(fdata);
    var r = fw.packed_model.Reader{ .data = fdata };
    const magic = r.take(MAGIC.len) orelse return error.Truncated;
    if (!std.mem.eql(u8, magic, MAGIC)) return error.BadMagic;
    const in_size = r.u32le() orelse return error.Truncated;
    const units_total = r.u32le() orelse return error.Truncated;
    const out_dim = r.u32le() orelse return error.Truncated;
    const n_layers = r.u32le() orelse return error.Truncated;
    const mode = r.u32le() orelse return error.Truncated;
    const bb_units = r.u32le() orelse return error.Truncated;
    const proj_size = r.u32le() orelse return error.Truncated;
    const cell_kind = r.u32le() orelse return error.Truncated;
    if (in_size != m.in_size or units_total != m.units_total or out_dim != m.out_dim) return error.ShapeMismatch;
    if (n_layers != m.layers.len or mode != @intFromEnum(m.mode)) return error.ShapeMismatch;
    if (bb_units != (if (m.bb != null) m.bb.?.n_out else 0)) return error.ShapeMismatch;
    if (proj_size != (if (m.proj != null) m.proj.?.n_out else 0)) return error.ShapeMismatch;
    if (cell_kind != (if (m.ltc != null) @as(u32, 1) else 0)) return error.ShapeMismatch;
    if (m.ltc) |ltc| {
        _ = try fw.packed_model.readMatrix(&r, ltc.w.w.len, ltc.w.qw.?, std.heap.page_allocator);
        _ = try fw.packed_model.readMatrix(&r, ltc.sensory_w.w.len, ltc.sensory_w.qw.?, std.heap.page_allocator);
        const fps = [13]*Param{ &ltc.sigma, &ltc.mu, &ltc.erev, &ltc.sensory_sigma, &ltc.sensory_mu, &ltc.sensory_erev, &ltc.gleak, &ltc.vleak, &ltc.cm, &ltc.input_w, &ltc.input_b, &ltc.output_w, &ltc.output_b };
        for (fps) |p| {
            const b = r.take(p.w.len * 4) orelse return error.Truncated;
            @memcpy(p.w, std.mem.bytesAsSlice(f32, b));
        }
    }
    if (m.bb != null) try load_lin(&r, &m.bb.?);
    for (m.layers) |*l| {
        const n_sens = r.u32le() orelse return error.Truncated;
        const n_out = r.u32le() orelse return error.Truncated;
        if (n_sens != l.n_sens or n_out != l.n_out) return error.ShapeMismatch;
        try load_lin(&r, &l.ff1);
        if (l.mode != .pure) {
            try load_lin(&r, &l.ff2.?);
            try load_lin(&r, &l.t_a.?);
            try load_lin(&r, &l.t_b.?);
        } else {
            const tau = r.take(n_out * 4) orelse return error.Truncated;
            @memcpy(l.tau.?.w, std.mem.bytesAsSlice(f32, tau));
            const aa = r.take(n_out * 4) orelse return error.Truncated;
            @memcpy(l.a_par.?.w, std.mem.bytesAsSlice(f32, aa));
        }
    }
    if (m.proj != null) try load_lin(&r, &m.proj.?);
    _ = r.f64le() orelse return error.Truncated; // expected mse, consumed by the C engine
}

fn load_lin(r: *fw.packed_model.Reader, l: *Lin) !void {
    _ = try fw.packed_model.readMatrix(r, l.p.w.len, l.p.qw.?, std.heap.page_allocator);
    const b = r.take(l.b.w.len * 4) orelse return error.Truncated;
    @memcpy(l.b.w, std.mem.bytesAsSlice(f32, b));
}

// ---------------------------------------------------------------------------
// benchmark + canonical-sequence demo
// ---------------------------------------------------------------------------

var bench_sink: f32 = 0;

fn bench_latency(m: *Model, ternary_weights: bool, iters: usize) f64 {
    const saved_wq = weights_q;
    const saved_aq = act_q;
    weights_q = ternary_weights;
    act_q = false;
    if (ternary_weights) ternarize_model(m);
    defer {
        weights_q = saved_wq;
        act_q = saved_aq;
    }
    var sink: f32 = 0;
    for (0..100) |i| {
        forward_seq(m, &seq_x[i % N_TRAIN]);
        sink += m.c_pred[0];
    }
    const t0 = fw.timing.nowS();
    for (0..iters) |i| {
        forward_seq(m, &seq_x[i % N_TRAIN]);
        sink += m.c_pred[0];
    }
    const elapsed = fw.timing.nowS() - t0;
    bench_sink += sink;
    return elapsed / @as(f64, @floatFromInt(iters)) * 1e6;
}

/// The exact ncps torch_cfc_sinusoidal sequence (k=1, phase 0, no noise):
/// sin/cos over [0, 3*pi] in, sin over [0, 6*pi] out.
var canon_x: [T_SEQ * IN_DIM]f32 = undefined;
var canon_y: [T_SEQ * OUT_DIM]f32 = undefined;

fn canonical_demo(m: *Model) f64 {
    const saved_wq = weights_q;
    const saved_aq = act_q;
    weights_q = true;
    act_q = false;
    defer {
        weights_q = saved_wq;
        act_q = saved_aq;
    }
    forward_seq(m, &canon_x);
    var ss: f64 = 0;
    for (0..T_SEQ * OUT_DIM) |i| {
        const d = m.c_pred[i] - canon_y[i];
        ss += d * d;
    }
    const mse = ss / @as(f64, T_SEQ * OUT_DIM);
    _ = clib.printf("  canonical ncps sequence (deployment weights): MSE %.6f\n", mse);
    _ = clib.printf("    t  target  pred  |");
    for (0..T_SEQ) |t| {
        if (t % 8 == 0) {
            _ = clib.printf(" t%02d %+.3f/%+.3f", @as(c_int, @intCast(t)), @as(f64, canon_y[t]), @as(f64, m.c_pred[t]));
        }
    }
    _ = clib.printf("\n");
    return mse;
}

// ---------------------------------------------------------------------------
// per-model pipeline
// ---------------------------------------------------------------------------

const Results = struct {
    fp32_val: f64 = 0,
    ptq_val: f64 = 0,
    qat_val: f64 = 0,
    qat_train: f64 = 0,
    packed_bytes: usize = 0,
    lat_fp32: f64 = 0,
    lat_tern: f64 = 0,
    roundtrip_ok: bool = false,
    canonical_mse: f64 = 0,
    n_params: usize = 0,
    n_ternary: usize = 0,
    hist: [3]usize = .{ 0, 0, 0 },
};

fn ternary_hist(m: *Model, hist: *[3]usize) void {
    hist.* = .{ 0, 0, 0 };
    for (m.plist) |p| {
        if (p.qw == null) continue;
        fw.quant.histTrits(p.w, trits_scratch[0..p.w.len], hist);
    }
}

fn run_pipeline(init: std.process.Init, alloc: std.mem.Allocator, m: *Model, fp32_steps: usize, qat_steps: usize, lr_fp32: f32, lr_qat: f32, batch: usize, bench_iters: usize, model_path: []const u8) Results {
    var res = Results{};
    _ = clib.printf("\n== model: %s ==\n", @as([*]const u8, @ptrCast(m.name.ptr)));
    var n_params: usize = 0;
    var n_ternary: usize = 0;
    for (m.plist) |p| {
        n_params += p.w.len;
        if (p.qw != null) n_ternary += p.w.len;
    }
    res.n_params = n_params;
    res.n_ternary = n_ternary;
    _ = clib.printf("  %d params (ternary %d, fp32 %d) | %d MACs/sequence\n", @as(c_int, @intCast(n_params)), @as(c_int, @intCast(n_ternary)), @as(c_int, @intCast(n_params - n_ternary)), @as(c_int, @intCast(m.macs_per_seq)));

    _ = clib.printf("\n== phase 1: fp32 pretrain (BPTT) ==\n");
    train_phase(m, false, fp32_steps, lr_fp32, batch);

    weights_q = false;
    const fp32_train = eval_mse(m, 0, N_TRAIN);
    res.fp32_val = eval_mse(m, N_TRAIN, N_VAL);
    _ = clib.printf("\neval (MSE, float activations)\n");
    _ = clib.printf("  fp32  train %.6f   val %.6f\n", fp32_train, res.fp32_val);

    // post-training quantization reference: ternarize and evaluate as-is
    ternarize_model(m);
    weights_q = true;
    res.ptq_val = eval_mse(m, N_TRAIN, N_VAL);
    _ = clib.printf("  ptq   val %.6f   (ternary, no QAT)\n", res.ptq_val);

    _ = clib.printf("\n== phase 2: BitNet ternary QAT ==\n");
    reset_adam(m);
    train_phase(m, true, qat_steps, lr_qat, batch);

    ternarize_model(m);
    weights_q = true;
    res.qat_train = eval_mse(m, 0, N_TRAIN);
    res.qat_val = eval_mse(m, N_TRAIN, N_VAL);
    _ = clib.printf("\neval (MSE, deployment tables, float activations)\n");
    _ = clib.printf("  qat   train %.6f   val %.6f\n", res.qat_train, res.qat_val);

    ternary_hist(m, &res.hist);
    const total: f64 = @floatFromInt(res.hist[0] + res.hist[1] + res.hist[2]);
    _ = clib.printf("  trits: -1 %.1f%%   0 %.1f%%   +1 %.1f%%\n", 100.0 * @as(f64, @floatFromInt(res.hist[0])) / total, 100.0 * @as(f64, @floatFromInt(res.hist[1])) / total, 100.0 * @as(f64, @floatFromInt(res.hist[2])) / total);

    res.lat_fp32 = bench_latency(m, false, bench_iters);
    res.lat_tern = bench_latency(m, true, bench_iters);
    _ = clib.printf("\nlatency (host CPU, %d-step sequence, float activations)\n", @as(c_int, @intCast(m.T)));
    _ = clib.printf("  fp32 weights         : %8.2f us/sequence  (%.0fM MACs/s)\n", res.lat_fp32, @as(f64, @floatFromInt(m.macs_per_seq)) / res.lat_fp32);
    _ = clib.printf("  ternary weights      : %8.2f us/sequence  (%.0fM MACs/s)\n", res.lat_tern, @as(f64, @floatFromInt(m.macs_per_seq)) / res.lat_tern);

    _ = clib.printf("\n== phase 3: packed export + round-trip ==\n");
    res.packed_bytes = export_model(init.io, alloc, m, model_path, res.qat_val) catch {
        std.debug.print("failed to write {s}\n", .{model_path});
        clib.exit(1);
    };
    _ = clib.printf("  wrote %s (%d bytes)\n", @as([*]const u8, @ptrCast(model_path.ptr)), @as(c_int, @intCast(res.packed_bytes)));
    _ = load_model(init.io, alloc, m, model_path) catch {
        std.debug.print("failed to load {s}\n", .{model_path});
        clib.exit(1);
    };
    const val_after = eval_mse(m, N_TRAIN, N_VAL);
    _ = clib.printf("  val MSE pre-export   : %.6f\n", res.qat_val);
    _ = clib.printf("  val MSE reloaded     : %.6f\n", val_after);
    res.roundtrip_ok = res.qat_val == val_after;
    if (res.roundtrip_ok) {
        _ = clib.printf("  round-trip: OK (lossless)\n");
    } else {
        _ = clib.printf("  round-trip: MISMATCH\n");
    }
    res.canonical_mse = canonical_demo(m);

    const fp32_bytes = n_params * 4;
    _ = clib.printf("\nstorage\n");
    _ = clib.printf("  fp32 weights         : %6d bytes\n", @as(c_int, @intCast(fp32_bytes)));
    _ = clib.printf("  naive 2-bit          : %6d bytes\n", @as(c_int, @intCast((n_ternary + 3) / 4 + (n_params - n_ternary) * 4)));
    _ = clib.printf("  packed 5-trits/byte  : %6d bytes  (%.1fx smaller than fp32)\n", @as(c_int, @intCast(res.packed_bytes)), @as(f64, @floatFromInt(fp32_bytes)) / @as(f64, @floatFromInt(res.packed_bytes)));

    const recovery = (1.0 - (res.qat_val - res.fp32_val) / (res.ptq_val - res.fp32_val + 1e-12)) * 100.0;
    _ = clib.printf("\nsummary\n");
    if (res.ptq_val - res.fp32_val > 1e-9) {
        _ = clib.printf("  fp32 val %.6f | ptq val %.6f | qat val %.6f | qat recovery %.1f%% of fp32->ptq MSE gap\n", res.fp32_val, res.ptq_val, res.qat_val, recovery);
    } else {
        // PTQ already lossless: there is no quantization gap to recover
        _ = clib.printf("  fp32 val %.6f | ptq val %.6f (lossless) | qat val %.6f | ternary w/o any accuracy cost\n", res.fp32_val, res.ptq_val, res.qat_val);
    }
    return res;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

pub fn main(init: std.process.Init) u8 {
    // this example's fixed RNG stream: identical runs on every machine
    rng.state = 0x4C1C_1E41_2026_0001;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const alloc = arena.allocator();

    _ = clib.printf("== example-ncps: Neural Circuit Policies (CfC/LTC/NCP), ternary pipeline ==\n");
    _ = clib.printf("task: sine-frequency doubling over %d steps (ncps torch_cfc_sinusoidal family)\n", @as(c_int, @intCast(T_SEQ)));
    _ = clib.printf("input [sin, cos] over [0, 3*pi*k] -> target sin over [0, 6*pi*k], MSE loss\n");

    gen_dataset();
    gen_sequence(1.0, 0.0, 0.0, &canon_x, &canon_y); // the ncps example sequence
    _ = clib.printf("dataset: %d sequences (train %d, val %d), freq k in [%.1f, %.1f], noise %.3f\n", @as(c_int, @intCast(N_SEQ)), @as(c_int, @intCast(N_TRAIN)), @as(c_int, @intCast(N_VAL)), 1.0 - FREQ_JITTER, 1.0 + FREQ_JITTER, @as(f64, NOISE_STD));

    // ncps.torch.CfC(2, 32, proj_size=1) with the default LeCun backbone
    const m1 = build_unwired(alloc, "CfC(2->32, backbone 64, proj 1)", IN_DIM, 32, 64, 1, .default);
    // ncps.torch.WiredCfC + wirings.FullyConnected(8, 1)
    const w2 = build_fully_connected(alloc, IN_DIM, 8, 1, 1111);
    const m2 = build_wired(alloc, "WiredCfC + FullyConnected(8, out 1)", &w2, .default);
    // ncps.torch.WiredCfC + wirings.NCP(16 inter, 8 command, 1 motor) -- the
    // sensory/inter/command/motor circuit from the ncps sinusoidal example
    const w3 = build_ncp(alloc, IN_DIM, 16, 8, 1, 12, 4, 5, 8, 22222);
    // ncps.torch.LTC on the same NCP wiring: the liquid ODE cell
    const m4 = build_ltc(alloc, "LTC + NCP(16 inter, 8 cmd, 1 motor)", &w3);
    _ = clib.printf("wirings: FullyConnected(8): %d synapses | NCP(16,8,1): %d units, %d synapses (sparse circuit)\n", @as(c_int, @intCast(w2.synapseCount())), @as(c_int, @intCast(w3.units)), @as(c_int, @intCast(w3.synapseCount())));
    const m3 = build_wired(alloc, "WiredCfC + NCP(16 inter, 8 cmd, 1 motor)", &w3, .default);

    var ok = true;
    const r1 = run_pipeline(init, alloc, m1, 1400, 1000, 5e-3, 1e-3, 16, 1500, "examples/liquid_sine/modeldata/ncps_cfc_packed.bin");
    const r2 = run_pipeline(init, alloc, m2, 3000, 2000, 5e-3, 1e-3, 16, 20000, "examples/liquid_sine/modeldata/ncps_fc_packed.bin");
    const r3 = run_pipeline(init, alloc, m3, 1500, 1500, 5e-3, 1e-3, 16, 20000, "examples/liquid_sine/modeldata/ncps_ncp_packed.bin");
    const r4 = run_pipeline(init, alloc, m4, 1500, 1500, 3e-3, 1e-3, 16, 3000, "examples/liquid_sine/modeldata/ncps_ltc_packed.bin");
    _ = export_valset(init.io, alloc, "examples/liquid_sine/modeldata/ncps_valset.bin") catch {
        std.debug.print("failed to write valset\n", .{});
        return 1;
    };
    _ = clib.printf("\nwrote examples/liquid_sine/modeldata/ncps_valset.bin (val split for the C engine)\n");

    ok = r1.roundtrip_ok and r2.roundtrip_ok and r3.roundtrip_ok and r4.roundtrip_ok;

    _ = clib.printf("\n== all models ==\n");
    _ = clib.printf("  %-42s %9s %9s %9s %8s %8s %8s\n", "model", "fp32 val", "ptq val", "qat val", "recovery", "packed", "us/seq");
    inline for (.{
        .{ m1.name, r1 },
        .{ m2.name, r2 },
        .{ m3.name, r3 },
        .{ m4.name, r4 },
    }) |row| {
        const name, const r = .{ row[0], row[1] };
        const gap = r.ptq_val - r.fp32_val;
        if (gap > 1e-9) {
            _ = clib.printf("  %-42s %9.6f %9.6f %9.6f %7.1f%% %6d B %8.2f\n", @as([*]const u8, @ptrCast(name.ptr)), r.fp32_val, r.ptq_val, r.qat_val, 100.0 * (1.0 - (r.qat_val - r.fp32_val) / gap), @as(c_int, @intCast(r.packed_bytes)), r.lat_tern);
        } else {
            _ = clib.printf("  %-42s %9.6f %9.6f %9.6f %8s %6d B %8.2f\n", @as([*]const u8, @ptrCast(name.ptr)), r.fp32_val, r.ptq_val, r.qat_val, "lossless", @as(c_int, @intCast(r.packed_bytes)), r.lat_tern);
        }
    }
    _ = clib.printf("  sink %.4f\n", @as(f64, bench_sink));

    return if (ok) 0 else 1;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

const FlagSave = struct { wq: bool, aq: bool };

fn set_flags(wq: bool, aq: bool) FlagSave {
    const old = FlagSave{ .wq = weights_q, .aq = act_q };
    weights_q = wq;
    act_q = aq;
    return old;
}

fn loss_of(m: *Model, x: []const f32, y: []const f32) f64 {
    forward_seq(m, x);
    var ss: f64 = 0;
    for (0..m.T * m.out_dim) |i| {
        const d = m.c_pred[i] - y[i];
        ss += d * d;
    }
    return ss;
}

/// Numeric-vs-analytic gradient check over every parameter (strided sample).
/// Masked weight coordinates are skipped: the wiring zeroes those latents at
/// init and blocks their gradients, so the deployed model has no degree of
/// freedom there -- perturbing them measures nothing.
fn check_grads(m: *Model, x: []const f32, y: []const f32, stride: usize, max_checks: usize) !void {
    zero_model_grads(m);
    forward_seq(m, x);
    backward_seq(m, x, y, 1.0);
    const h: f32 = 1e-3;
    for (m.plist, 0..) |p, pi| {
        const mask = if (pi < m.pmask.len) m.pmask[pi] else null;
        var idx: usize = 0;
        var checked: usize = 0;
        while (idx < p.w.len and checked < max_checks) : (idx += stride) {
            if (mask) |mm| {
                if (mm[idx] == 0) continue;
            }
            const saved = p.w[idx];
            p.w[idx] = saved + h;
            const lp = loss_of(m, x, y);
            p.w[idx] = saved - h;
            const lm = loss_of(m, x, y);
            p.w[idx] = saved;
            const num = (lp - lm) / (2 * h);
            if (@abs(num - p.d[idx]) > 1e-3 + 3e-2 * @abs(num)) {
                std.debug.print("  grad mismatch: param #{d} idx {d}/{d}: analytic {d}, numeric {d}\n", .{ pi, idx, p.w.len, p.d[idx], num });
            }
            try testing.expectApproxEqAbs(num, p.d[idx], 1e-3 + 3e-2 * @abs(num));
            checked += 1;
        }
    }
}

test "unwired CfC BPTT gradients match finite differences (default mode)" {
    const saved_rng = rng.state;
    defer rng.state = saved_rng;
    rng.state = 0xA11CE_0001;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const T = 6;
    const m = build_unwired(alloc, "test", 2, 4, 5, 1, .default);
    m.T = T;
    alloc_caches(alloc, m);
    const old = set_flags(false, false);
    defer _ = set_flags(old.wq, old.aq);

    var x: [T * 2]f32 = undefined;
    var y: [T * 1]f32 = undefined;
    for (&x) |*v| v.* = rng.gauss(0, 0.5);
    for (&y) |*v| v.* = rng.gauss(0, 0.5);
    try check_grads(m, &x, &y, 7, 6);
}

test "unwired CfC BPTT gradients match finite differences (pure + no_gate modes)" {
    const saved_rng = rng.state;
    defer rng.state = saved_rng;
    rng.state = 0xA11CE_0002;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    inline for (.{ Mode.pure, Mode.no_gate }) |mode| {
        const T = 5;
        const m = build_unwired(alloc, "test", 2, 4, 0, 1, mode);
        m.T = T;
        alloc_caches(alloc, m);
        const old = set_flags(false, false);
        defer _ = set_flags(old.wq, old.aq);

        var x: [T * 2]f32 = undefined;
        var y: [T * 1]f32 = undefined;
        for (&x) |*v| v.* = rng.gauss(0, 0.5);
        for (&y) |*v| v.* = rng.gauss(0, 0.5);
        try check_grads(m, &x, &y, 5, 5);
    }
}

test "wired NCP layer stack gradients match finite differences" {
    const saved_rng = rng.state;
    defer rng.state = saved_rng;
    rng.state = 0xA11CE_0003;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const T = 5;
    const w = build_ncp(alloc, 2, 4, 3, 1, 2, 2, 3, 2, 7);
    const m = build_wired(alloc, "test ncp", &w, .default);
    m.T = T;
    alloc_caches(alloc, m);
    const old = set_flags(false, false);
    defer _ = set_flags(old.wq, old.aq);

    var x: [T * 2]f32 = undefined;
    var y: [T * 1]f32 = undefined;
    for (&x) |*v| v.* = rng.gauss(0, 0.5);
    for (&y) |*v| v.* = rng.gauss(0, 0.5);
    try check_grads(m, &x, &y, 11, 6);
}

test "LTC ODE-cell gradients match finite differences (BPTT through the solver)" {
    const saved_rng = rng.state;
    defer rng.state = saved_rng;
    rng.state = 0xA11CE_0008;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const T = 4;
    const w = build_ncp(alloc, 2, 4, 3, 1, 2, 2, 3, 2, 7);
    const m = build_ltc(alloc, "test ltc", &w);
    m.T = T;
    alloc_caches(alloc, m);
    const old = set_flags(false, false);
    defer _ = set_flags(old.wq, old.aq);

    var x: [T * 2]f32 = undefined;
    var y: [T * 1]f32 = undefined;
    for (&x) |*v| v.* = rng.gauss(0, 0.5);
    for (&y) |*v| v.* = rng.gauss(0, 0.5);
    // smaller tolerance band than CfC: the ODE quotient rule is exact, so
    // only float roundoff through 4 unfolds x 4 timesteps remains
    try check_grads(m, &x, &y, 7, 5);
}

test "LTC masked synapse gradients stay zero" {
    const saved_rng = rng.state;
    defer rng.state = saved_rng;
    rng.state = 0xA11CE_0009;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const w = build_ncp(alloc, 2, 6, 3, 1, 2, 2, 2, 2, 9);
    const m = build_ltc(alloc, "test ltc mask", &w);
    m.T = 3;
    alloc_caches(alloc, m);
    const old = set_flags(false, false);
    defer _ = set_flags(old.wq, old.aq);
    var x: [3 * 2]f32 = undefined;
    var y: [3 * 1]f32 = undefined;
    for (&x) |*v| v.* = rng.gauss(0, 0.5);
    for (&y) |*v| v.* = rng.gauss(0, 0.5);
    zero_model_grads(m);
    forward_seq(m, &x);
    backward_seq(m, &x, &y, 1.0);
    const ltc = m.ltc.?;
    for (ltc.mask, ltc.w.d) |mm, gg| {
        if (mm == 0) try testing.expectEqual(@as(f32, 0), gg);
    }
    for (ltc.sensory_mask, ltc.sensory_w.d) |mm, gg| {
        if (mm == 0) try testing.expectEqual(@as(f32, 0), gg);
    }
}

test "wiring mask zeroes masked latents and blocks their gradients" {
    const saved_rng = rng.state;
    defer rng.state = saved_rng;
    rng.state = 0xA11CE_0004;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const w = build_ncp(alloc, 2, 6, 3, 1, 2, 2, 2, 2, 9);
    const m = build_wired(alloc, "test mask", &w, .default);
    // every masked latent must have been zeroed at init
    for (m.layers) |*l| {
        const mask = l.mask.?;
        try testing.expectEqual(l.ff1.p.w.len, mask.len);
        for (mask, l.ff1.p.w) |mm, ww| {
            if (mm == 0) try testing.expectEqual(@as(f32, 0), ww);
        }
    }
    // and one backward pass must leave masked gradient entries at zero
    m.T = 4;
    alloc_caches(alloc, m);
    const old = set_flags(false, false);
    defer _ = set_flags(old.wq, old.aq);
    var x: [4 * 2]f32 = undefined;
    var y: [4 * 1]f32 = undefined;
    for (&x) |*v| v.* = rng.gauss(0, 0.5);
    for (&y) |*v| v.* = rng.gauss(0, 0.5);
    zero_model_grads(m);
    forward_seq(m, &x);
    backward_seq(m, &x, &y, 1.0);
    for (m.layers) |*l| {
        const mask = l.mask.?;
        for (mask, l.ff1.p.d) |mm, gg| {
            if (mm == 0) try testing.expectEqual(@as(f32, 0), gg);
        }
    }
}

test "AutoNCP derivation matches the reference formula" {
    const saved_rng = rng.state;
    defer rng.state = saved_rng;
    rng.state = 0xA11CE_0005;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // AutoNCP(units=20, output_size=1, sparsity_level=0.5)
    const w = build_autoncp(alloc, 2, 20, 1, 0.5, 22222);
    try testing.expectEqual(@as(usize, 20), w.units);
    try testing.expectEqual(@as(usize, 1), w.output_dim);
    // inter_and_command = 19, command = max(int(0.4*19),1) = 7, inter = 12
    try testing.expectEqual(@as(usize, 12), w.id_len[0]);
    try testing.expectEqual(@as(usize, 7), w.id_len[1]);
    try testing.expectEqual(@as(usize, 1), w.id_len[2]);
    // state layout: inter, command, motor
    try testing.expectEqual(@as(usize, 0), w.slice[0].off);
    try testing.expectEqual(@as(usize, 12), w.slice[1].off);
    try testing.expectEqual(@as(usize, 19), w.slice[2].off);
    // density 0.5: sensory_fanout 6, inter_fanout 3, recurrent 7, motor_fanin 3.
    // Each sensory neuron gets exactly sensory_fanout synapses from the main
    // pass; the reachability repair below may add more.
    var sensory_out = [_]usize{0} ** 2;
    for (0..2) |s| {
        for (0..w.units) |d| sensory_out[s] += @intCast(@abs(w.sensory_adj[s * w.units + d]));
    }
    for (sensory_out) |n| try testing.expect(n >= 6);
}

test "NCP wiring structure: reachability and fan-outs" {
    const saved_rng = rng.state;
    defer rng.state = saved_rng;
    rng.state = 0xA11CE_0006;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const w = build_ncp(alloc, 2, 16, 8, 1, 12, 4, 5, 8, 22222);
    try testing.expectEqual(@as(usize, 25), w.units);
    // every sensory neuron has at least sensory_fanout outgoing synapses
    // (the main pass gives exactly sensory_fanout; repair can add more)
    for (0..2) |s| {
        var n: usize = 0;
        for (0..w.units) |d| n += @intCast(@abs(w.sensory_adj[s * w.units + d]));
        try testing.expect(n >= 12);
    }
    // every inter neuron has exactly inter_fanout outgoing synapses
    for (w.id_start[0]..w.id_start[0] + w.id_len[0]) |src| {
        var n: usize = 0;
        for (0..w.units) |d| n += @intCast(@abs(w.adjacency[src * w.units + d]));
        try testing.expectEqual(@as(usize, 4), n);
    }
    // every command neuron receives from at least one inter neuron ...
    for (w.id_start[1]..w.id_start[1] + w.id_len[1]) |dest| {
        var n: usize = 0;
        for (w.id_start[0]..w.id_start[0] + w.id_len[0]) |src| {
            n += @intCast(@abs(w.adjacency[src * w.units + dest]));
        }
        try testing.expect(n >= 1);
    }
    // ... and every motor neuron has exactly motor_fanin incoming synapses
    for (0..w.output_dim) |dest| {
        var n: usize = 0;
        for (w.id_start[1]..w.id_start[1] + w.id_len[1]) |src| {
            n += @intCast(@abs(w.adjacency[src * w.units + dest]));
        }
        try testing.expectEqual(@as(usize, 8), n);
    }
    // the mask reproduces the wiring: layer-2 ff1 sees exactly the wired
    // command synapses plus the dense recurrent block
    const m = build_wired(alloc, "test", &w, .default);
    const l2 = &m.layers[2];
    const mask = l2.mask.?;
    try testing.expectEqual(@as(usize, 1), l2.n_out);
    try testing.expectEqual(@as(usize, 9), l2.cat_in);
    var wired: usize = 0;
    for (0..8) |i| wired += @intFromFloat(mask[i]);
    try testing.expectEqual(@as(usize, 8), wired); // motor_fanin synapses
    for (8..9) |i| try testing.expectEqual(@as(f32, 1), mask[i]); // recurrent
}

test "FullyConnected wiring is dense with ±1 polarities" {
    const saved_rng = rng.state;
    defer rng.state = saved_rng;
    rng.state = 0xA11CE_0007;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const w = build_fully_connected(alloc, 2, 8, 1, 1111);
    try testing.expectEqual(@as(usize, 1), w.n_layers);
    for (w.adjacency) |a| try testing.expect(a == 1 or a == -1);
    for (w.sensory_adj) |a| try testing.expect(a == 1 or a == -1);
    var plus: usize = 0;
    for (w.adjacency) |a| {
        if (a == 1) plus += 1;
    }
    // choice([-1,1,1]): about 2/3 plus, statistically bounded away from 1/2
    try testing.expect(plus * 2 > w.adjacency.len);
}

test "dataset is deterministic and the canonical sequence matches the ncps example" {
    const saved_rng = rng.state;
    defer rng.state = saved_rng;

    rng.state = 0xBEEF;
    var a: [T_SEQ * IN_DIM]f32 = undefined;
    var b: [T_SEQ * OUT_DIM]f32 = undefined;
    gen_sequence(1.3, 0.7, 0.02, &a, &b);
    rng.state = 0xBEEF;
    var a2: [T_SEQ * IN_DIM]f32 = undefined;
    var b2: [T_SEQ * OUT_DIM]f32 = undefined;
    gen_sequence(1.3, 0.7, 0.02, &a2, &b2);
    try testing.expectEqualSlices(f32, &a, &a2);
    try testing.expectEqualSlices(f32, &b, &b2);

    // k=1, phase 0, no noise is the ncps example: x[0] = (0, 1), y[0] = 0,
    // y[T-1] = sin(6*pi) = 0, x[T-1] = (sin 3pi, cos 3pi) = (~0, -1)
    gen_sequence(1.0, 0.0, 0.0, &a, &b);
    try testing.expectApproxEqAbs(@as(f32, 0), a[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1), a[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), b[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), b[T_SEQ - 1], 1e-5);
}

test "ternary QAT forward stays close to the fp32 forward (trained weights are not required)" {
    const saved_rng = rng.state;
    defer rng.state = saved_rng;
    rng.state = 0x1DCA_0002;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const m = build_unwired(alloc, "test", 2, 8, 8, 1, .default);
    m.T = 4;
    alloc_caches(alloc, m);
    var x: [4 * 2]f32 = undefined;
    for (&x) |*v| v.* = rng.gauss(0, 0.5);
    const old = set_flags(false, false);
    defer _ = set_flags(old.wq, old.aq);

    forward_seq(m, &x);
    var fp32_pred: [4]f32 = undefined;
    @memcpy(&fp32_pred, m.c_pred[0..4]);

    ternarize_model(m);
    _ = set_flags(true, false);
    forward_seq(m, &x);
    var max_shift: f32 = 0;
    for (0..4) |i| max_shift = @max(max_shift, @abs(fp32_pred[i] - m.c_pred[i]));
    // random untrained weights tolerate a loose bound; the real gate is the
    // fp32/ptq/qat comparison in the pipeline
    try testing.expect(max_shift < 1.0);
}
