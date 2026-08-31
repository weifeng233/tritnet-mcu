//! Full-pipeline BitNet GPT in pure Zig.
//!
//! Pipeline (single binary, no dependencies beyond libc):
//!   1. fp32 pretrain        -- identical math to src/microgpt.zig
//!   2. BitNet QAT           -- ternary {-1,0,+1} weights via STE, absmean
//!                              per-tensor scales, per-token 8-bit activations
//!   3. validation           -- held-out loss for fp32 / PTQ / QAT, samples
//!   4. ternary packing      -- 5 trits per byte (compilade.net scheme), the
//!                              same packing llama.cpp uses for BitNet weights
//!   5. round-trip check     -- reload the packed file, unpack, re-evaluate;
//!                              the loss must match the pre-export model exactly
//!
//! Weight quantizer (BitNet b1.58 style):
//!   scale = mean(|W|) per tensor
//!   trit  = W > scale/2 ? +1 : W < -scale/2 ? -1 : 0
//!   deployed weight = scale * trit
//! Backward uses the straight-through estimator with the gradient masked to
//! |W| < scale (mirrors bitnet_gpt_train.py).

const std = @import("std");
const builtin = @import("builtin");
const math = std.math;
const ternary = @import("ternary.zig");

const clib = @cImport({
    @cInclude("stdio.h");
    @cInclude("stdlib.h");
    @cInclude("string.h");
    @cInclude("math.h");
});
const winclib = if (builtin.os.tag == .windows) @cImport({
    @cInclude("windows.h");
}) else struct {};
const timeclib = if (builtin.os.tag != .windows) @cImport({
    @cInclude("time.h");
}) else struct {};

const is_x86 = builtin.cpu.arch == .x86_64;
const is_aarch64 = builtin.cpu.arch == .aarch64;

const MAX_DOCS = 85000;
const MAX_DOC_LEN = 512;
const MAX_CHARS = 128;

const N_EMBD: usize = 16;
const N_HEAD: usize = 4;
const N_LAYER: usize = 1;
const BLOCK_SIZE: usize = 16;
const HEAD_DIM: usize = N_EMBD / N_HEAD;
const MLP_DIM: usize = 4 * N_EMBD;

const LM_PAD_MAX: usize = ((MAX_CHARS + 1 + 3) / 4) * 4;
const ATTN_SCALE: f32 = 0.5;

const MAGIC = "MGPTQ001";

var docs: [MAX_DOCS][MAX_DOC_LEN]u8 = undefined;
var num_docs: usize = 0;
var n_train: usize = 0;

var uchars_arr: [MAX_CHARS]u8 = undefined;
var num_uchars: usize = 0;
var vocab_size: usize = 0;
var BOS: usize = 0;
var lm_pad: usize = LM_PAD_MAX;

var rng_state: u64 = 42;

inline fn rng_next() u64 {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

inline fn rng_uniform() f64 {
    return @as(f64, @floatFromInt(rng_next() >> 11)) * (1.0 / 9007199254740992.0);
}

inline fn rng_gauss(mean: f32, stddev: f32) f32 {
    var uu1 = rng_uniform();
    const uu2 = rng_uniform();
    if (uu1 < 1e-30) uu1 = 1e-30;
    return mean + stddev * @as(f32, @floatCast(math.sqrt(-2.0 * @log(uu1)) * @cos(2.0 * math.pi * uu2)));
}

fn shuffle_ints(arr: []i32) void {
    var i: usize = arr.len;
    while (i > 1) {
        i -= 1;
        const j: usize = @intFromFloat(rng_uniform() * @as(f64, @floatFromInt(i + 1)));
        const tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

fn load_dataset(filename: []const u8) void {
    var c_path: [1024]u8 = undefined;
    const len = @min(filename.len, c_path.len - 1);
    @memcpy(c_path[0..len], filename[0..len]);
    c_path[len] = 0;
    const f = clib.fopen(@ptrCast(&c_path), "r");
    if (f == null) {
        std.debug.print("Cannot open {s}\n", .{filename});
        clib.exit(1);
    }
    var line: [512]u8 = undefined;
    while (clib.fgets(@ptrCast(&line), 512, f) != null) {
        if (num_docs >= MAX_DOCS) break;
        var l: usize = clib.strlen(@ptrCast(&line));
        while (l > 0 and (line[l - 1] == '\n' or line[l - 1] == '\r')) {
            l -= 1;
        }
        line[l] = 0;
        if (l == 0) continue;
        const copy_len = @min(l, MAX_DOC_LEN - 1);
        @memcpy(docs[num_docs][0..copy_len], line[0..copy_len]);
        docs[num_docs][copy_len] = 0;
        @memset(docs[num_docs][copy_len + 1 ..], 0);
        num_docs += 1;
    }
    _ = clib.fclose(f);
}

var char_lut: [256]i32 = [_]i32{-1} ** 256;

fn char_to_id(ch: u8) i32 {
    return char_lut[ch];
}

fn cmp_char(_: void, a: u8, b: u8) bool {
    return a < b;
}

fn build_tokenizer() void {
    var seen: [256]bool = [_]bool{false} ** 256;
    for (0..num_docs) |d| {
        var i: usize = 0;
        while (docs[d][i] != 0) : (i += 1) {
            seen[docs[d][i]] = true;
        }
    }
    num_uchars = 0;
    for (0..256) |i| {
        if (seen[i]) {
            uchars_arr[num_uchars] = @intCast(i);
            num_uchars += 1;
        }
    }
    std.mem.sort(u8, uchars_arr[0..num_uchars], {}, cmp_char);
    BOS = num_uchars;
    vocab_size = num_uchars + 1;
    for (0..num_uchars) |i| char_lut[uchars_arr[i]] = @intCast(i);
}

// ---------------------------------------------------------------------------
// parameters
// ---------------------------------------------------------------------------

var wte: []f32 = undefined;
var d_wte: []f32 = undefined;
var wpe: []f32 = undefined;
var d_wpe: []f32 = undefined;
var lm_head: []f32 = undefined;
var d_lm_head: []f32 = undefined;

var attn_wq: [N_LAYER][]f32 = undefined;
var d_attn_wq: [N_LAYER][]f32 = undefined;
var attn_wk: [N_LAYER][]f32 = undefined;
var d_attn_wk: [N_LAYER][]f32 = undefined;
var attn_wv: [N_LAYER][]f32 = undefined;
var d_attn_wv: [N_LAYER][]f32 = undefined;
var attn_wo: [N_LAYER][]f32 = undefined;
var d_attn_wo: [N_LAYER][]f32 = undefined;
var mlp_fc1: [N_LAYER][]f32 = undefined;
var d_mlp_fc1: [N_LAYER][]f32 = undefined;
var mlp_fc2: [N_LAYER][]f32 = undefined;
var d_mlp_fc2: [N_LAYER][]f32 = undefined;

var adam_m_wte: []f32 = undefined;
var adam_v_wte: []f32 = undefined;
var adam_m_wpe: []f32 = undefined;
var adam_v_wpe: []f32 = undefined;
var adam_m_lm: []f32 = undefined;
var adam_v_lm: []f32 = undefined;
var adam_m_wq: [N_LAYER][]f32 = undefined;
var adam_v_wq: [N_LAYER][]f32 = undefined;
var adam_m_wk: [N_LAYER][]f32 = undefined;
var adam_v_wk: [N_LAYER][]f32 = undefined;
var adam_m_wv: [N_LAYER][]f32 = undefined;
var adam_v_wv: [N_LAYER][]f32 = undefined;
var adam_m_wo: [N_LAYER][]f32 = undefined;
var adam_v_wo: [N_LAYER][]f32 = undefined;
var adam_m_fc1: [N_LAYER][]f32 = undefined;
var adam_v_fc1: [N_LAYER][]f32 = undefined;
var adam_m_fc2: [N_LAYER][]f32 = undefined;
var adam_v_fc2: [N_LAYER][]f32 = undefined;

// quantized (ternary) weight tables: scale * trit, same shapes as above
var qw_q: [N_LAYER][N_EMBD * N_EMBD]f32 align(64) = [_][N_EMBD * N_EMBD]f32{[_]f32{0} ** (N_EMBD * N_EMBD)} ** N_LAYER;
var qw_k: [N_LAYER][N_EMBD * N_EMBD]f32 align(64) = [_][N_EMBD * N_EMBD]f32{[_]f32{0} ** (N_EMBD * N_EMBD)} ** N_LAYER;
var qw_v: [N_LAYER][N_EMBD * N_EMBD]f32 align(64) = [_][N_EMBD * N_EMBD]f32{[_]f32{0} ** (N_EMBD * N_EMBD)} ** N_LAYER;
var qw_o: [N_LAYER][N_EMBD * N_EMBD]f32 align(64) = [_][N_EMBD * N_EMBD]f32{[_]f32{0} ** (N_EMBD * N_EMBD)} ** N_LAYER;
var qw_fc1: [N_LAYER][N_EMBD * MLP_DIM]f32 align(64) = [_][N_EMBD * MLP_DIM]f32{[_]f32{0} ** (N_EMBD * MLP_DIM)} ** N_LAYER;
var qw_fc2: [N_LAYER][MLP_DIM * N_EMBD]f32 align(64) = [_][MLP_DIM * N_EMBD]f32{[_]f32{0} ** (MLP_DIM * N_EMBD)} ** N_LAYER;
var qw_lm: []f32 = undefined;

// per-tensor STE thresholds from the most recent ternarize
var tern_t: [N_LAYER][6]f32 = [_][6]f32{[_]f32{0} ** 6} ** N_LAYER;
var tern_t_lm: f32 = 0;

var num_params: usize = 0;
var gpa = std.heap.c_allocator;

fn make_param(size: usize, stddev: f32) []f32 {
    const p = gpa.alloc(f32, size) catch {
        std.debug.print("out of memory\n", .{});
        clib.exit(1);
        unreachable;
    };
    for (0..size) |i| p[i] = rng_gauss(0, stddev);
    num_params += size;
    return p;
}

fn make_zero(size: usize) []f32 {
    const p = gpa.alloc(f32, size) catch {
        std.debug.print("out of memory\n", .{});
        clib.exit(1);
        unreachable;
    };
    @memset(p, 0);
    return p;
}

fn init_params() void {
    const es = vocab_size * N_EMBD;
    const ps = BLOCK_SIZE * N_EMBD;
    const as = N_EMBD * N_EMBD;
    const ms = MLP_DIM * N_EMBD;
    wte = make_param(es, 0.02);
    d_wte = make_zero(es);
    adam_m_wte = make_zero(es);
    adam_v_wte = make_zero(es);
    wpe = make_param(ps, 0.02);
    d_wpe = make_zero(ps);
    adam_m_wpe = make_zero(ps);
    adam_v_wpe = make_zero(ps);
    lm_head = make_param(es, 0.02);
    d_lm_head = make_zero(es);
    adam_m_lm = make_zero(es);
    adam_v_lm = make_zero(es);
    qw_lm = make_zero(es);
    for (0..N_LAYER) |i| {
        attn_wq[i] = make_param(as, 0.02);
        d_attn_wq[i] = make_zero(as);
        adam_m_wq[i] = make_zero(as);
        adam_v_wq[i] = make_zero(as);
        attn_wk[i] = make_param(as, 0.02);
        d_attn_wk[i] = make_zero(as);
        adam_m_wk[i] = make_zero(as);
        adam_v_wk[i] = make_zero(as);
        attn_wv[i] = make_param(as, 0.02);
        d_attn_wv[i] = make_zero(as);
        adam_m_wv[i] = make_zero(as);
        adam_v_wv[i] = make_zero(as);
        attn_wo[i] = make_param(as, 0.0);
        d_attn_wo[i] = make_zero(as);
        adam_m_wo[i] = make_zero(as);
        adam_v_wo[i] = make_zero(as);
        mlp_fc1[i] = make_param(ms, 0.02);
        d_mlp_fc1[i] = make_zero(ms);
        adam_m_fc1[i] = make_zero(ms);
        adam_v_fc1[i] = make_zero(ms);
        mlp_fc2[i] = make_param(ms, 0.0);
        d_mlp_fc2[i] = make_zero(ms);
        adam_m_fc2[i] = make_zero(ms);
        adam_v_fc2[i] = make_zero(ms);
    }
    _ = clib.printf("num params: %d\n", @as(c_int, @intCast(num_params)));
}

// ---------------------------------------------------------------------------
// BitNet ternary quantization
// ---------------------------------------------------------------------------

const TernInfo = struct { t: f32, scale: f32 };

var trits_scratch: [MLP_DIM * N_EMBD]i8 = undefined;

/// Quantize a weight matrix to trits with an absmean scale.
fn quant_trits(w: []const f32, trits: []i8) TernInfo {
    var sum: f32 = 0;
    for (w) |x| sum += @abs(x);
    const scale = sum / @as(f32, @floatFromInt(w.len));
    const t = 0.5 * scale;
    for (w, trits) |x, *q| {
        q.* = if (x > t) 1 else if (x < -t) -1 else 0;
    }
    return .{ .t = t, .scale = scale };
}

/// Fill the quantized weight table (scale * trit) for one matrix, return t.
fn ternarize(w: []const f32, qw: []f32) f32 {
    const info = quant_trits(w, trits_scratch[0..w.len]);
    for (qw, trits_scratch[0..w.len]) |*q, tr| {
        q.* = @as(f32, @floatFromInt(tr)) * info.scale;
    }
    return info.t;
}

fn ternarize_all() void {
    for (0..N_LAYER) |li| {
        tern_t[li][0] = ternarize(attn_wq[li], qw_q[li][0..]);
        tern_t[li][1] = ternarize(attn_wk[li], qw_k[li][0..]);
        tern_t[li][2] = ternarize(attn_wv[li], qw_v[li][0..]);
        tern_t[li][3] = ternarize(attn_wo[li], qw_o[li][0..]);
        tern_t[li][4] = ternarize(mlp_fc1[li], qw_fc1[li][0..]);
        tern_t[li][5] = ternarize(mlp_fc2[li], qw_fc2[li][0..]);
    }
    tern_t_lm = ternarize(lm_head, qw_lm);
}

/// Straight-through estimator mask: zero gradients where |W| >= 2*t,
/// matching bitnet_gpt_train.py's `mask = |w| < 2*threshold`.
fn mask_grads(w: []const f32, d: []f32, t: f32) void {
    for (w, d) |x, *g| {
        if (!(@abs(x) < 2 * t)) g.* = 0;
    }
}

fn mask_all_grads() void {
    for (0..N_LAYER) |li| {
        mask_grads(attn_wq[li], d_attn_wq[li], tern_t[li][0]);
        mask_grads(attn_wk[li], d_attn_wk[li], tern_t[li][1]);
        mask_grads(attn_wv[li], d_attn_wv[li], tern_t[li][2]);
        mask_grads(attn_wo[li], d_attn_wo[li], tern_t[li][3]);
        mask_grads(mlp_fc1[li], d_mlp_fc1[li], tern_t[li][4]);
        mask_grads(mlp_fc2[li], d_mlp_fc2[li], tern_t[li][5]);
    }
    mask_grads(lm_head, d_lm_head, tern_t_lm);
}

// forward/backward mode flags:
//   weights_q -- use the quantized tables (QAT training + ternary eval/infer)
//   act_q     -- quantize activations per token to 8 bit (QAT training only;
//                deployment inference keeps float activations)
var weights_q = false;
var act_q = false;

inline fn TW(w: []f32, qw: []const f32) []const f32 {
    return if (weights_q) qw else w;
}

inline fn act_quant8(v: []f32) void {
    var m: f32 = 1e-5;
    for (v) |x| {
        const a = @abs(x);
        if (a > m) m = a;
    }
    const s: f32 = 127.0 / m;
    for (v) |*x| {
        // |x| <= m so the round lands inside int8 range; STE backward is identity
        x.* = @round(x.* * s) / s;
    }
}

inline fn AQ(v: []f32) void {
    if (act_q) act_quant8(v);
}

// ---------------------------------------------------------------------------
// forward / backward (shared by fp32 and QAT via the mode flags)
// ---------------------------------------------------------------------------

const PosActs = struct {
    x_embed: [N_EMBD]f32 = [_]f32{0} ** N_EMBD,
    rms_scale_init: f32 = 0,
    x_in: [N_LAYER][N_EMBD]f32 = [_][N_EMBD]f32{[_]f32{0} ** N_EMBD} ** N_LAYER,
    xn_attn: [N_LAYER][N_EMBD]f32 = [_][N_EMBD]f32{[_]f32{0} ** N_EMBD} ** N_LAYER,
    rms_scale_attn: [N_LAYER]f32 = [_]f32{0} ** N_LAYER,
    q: [N_LAYER][N_EMBD]f32 = [_][N_EMBD]f32{[_]f32{0} ** N_EMBD} ** N_LAYER,
    aw: [N_LAYER][N_HEAD][BLOCK_SIZE]f32 = [_][N_HEAD][BLOCK_SIZE]f32{[_][BLOCK_SIZE]f32{[_]f32{0} ** BLOCK_SIZE} ** N_HEAD} ** N_LAYER,
    attn_out: [N_LAYER][N_EMBD]f32 = [_][N_EMBD]f32{[_]f32{0} ** N_EMBD} ** N_LAYER,
    x_mid: [N_LAYER][N_EMBD]f32 = [_][N_EMBD]f32{[_]f32{0} ** N_EMBD} ** N_LAYER,
    xn_mlp: [N_LAYER][N_EMBD]f32 = [_][N_EMBD]f32{[_]f32{0} ** N_EMBD} ** N_LAYER,
    rms_scale_mlp: [N_LAYER]f32 = [_]f32{0} ** N_LAYER,
    mlp_pre: [N_LAYER][MLP_DIM]f32 = [_][MLP_DIM]f32{[_]f32{0} ** MLP_DIM} ** N_LAYER,
    mlp_post: [N_LAYER][MLP_DIM]f32 = [_][MLP_DIM]f32{[_]f32{0} ** MLP_DIM} ** N_LAYER,
    x_out: [N_EMBD]f32 = [_]f32{0} ** N_EMBD,
};

var saved: [BLOCK_SIZE]PosActs = [_]PosActs{.{}} ** BLOCK_SIZE;
var saved_probs: [BLOCK_SIZE][MAX_CHARS + 1]f32 = [_][MAX_CHARS + 1]f32{[_]f32{0} ** (MAX_CHARS + 1)} ** BLOCK_SIZE;

var kv_keys: [N_LAYER][BLOCK_SIZE][N_EMBD]f32 = undefined;
var kv_vals: [N_LAYER][BLOCK_SIZE][N_EMBD]f32 = undefined;
var dk_accum: [N_LAYER][BLOCK_SIZE][N_EMBD]f32 = undefined;
var dv_accum: [N_LAYER][BLOCK_SIZE][N_EMBD]f32 = undefined;

inline fn fexpf(x: f32) f32 {
    const bits: i32 = @as(i32, @intFromFloat(12102203.1615614 * x * 1.4426950408)) + 1065353216;
    return @bitCast(bits);
}

const Vec4 = @Vector(4, f32);
const Vec8 = @Vector(8, f32);

inline fn dot16(a: *const [16]f32, b: *const [16]f32) f32 {
    const va0: Vec8 = a[0..8].*;
    const vb0: Vec8 = b[0..8].*;
    const va1: Vec8 = a[8..16].*;
    const vb1: Vec8 = b[8..16].*;
    const p0 = va0 * vb0;
    const p1 = va1 * vb1;
    const s0 = p0 + p1;
    return @reduce(.Add, s0);
}

inline fn dot4(a: *const [4]f32, b: *const [4]f32) f32 {
    const va: Vec4 = a.*;
    const vb: Vec4 = b.*;
    return @reduce(.Add, va * vb);
}

inline fn dot64(a: [*c]const f32, b: *const [64]f32) f32 {
    var acc: Vec8 = @splat(0);
    inline for (0..8) |i| {
        const off = i * 8;
        const va: Vec8 = @as(*const [8]f32, @ptrCast(a + off)).*;
        const vb: Vec8 = @as(*const [8]f32, @ptrCast(b[off..][0..8])).*;
        acc = @mulAdd(Vec8, va, vb, acc);
    }
    return @reduce(.Add, acc);
}

inline fn linear_fwd(x: *const [N_EMBD]f32, w: []const f32, nout: usize, nin: usize, out: []f32) void {
    if (nin == N_EMBD and nout == N_EMBD) {
        for (0..16) |r| {
            const row: *const [16]f32 = @ptrCast(w.ptr + r * 16);
            out[r] = dot16(row, x);
        }
    } else if (nin == N_EMBD and nout == MLP_DIM) {
        for (0..64) |r| {
            const row: *const [16]f32 = @ptrCast(w.ptr + r * 16);
            out[r] = dot16(row, x);
        }
    } else if (nin == MLP_DIM and nout == N_EMBD) {
        for (0..16) |r| {
            out[r] = dot64(w.ptr + r * 64, @ptrCast(x));
        }
    } else {
        for (0..nout) |r| {
            const row: *const [16]f32 = @ptrCast(w.ptr + r * 16);
            out[r] = dot16(row, x);
        }
    }
}

inline fn rmsnorm_fwd(x: *const [N_EMBD]f32, n: usize, out: *[N_EMBD]f32) f32 {
    const v0: Vec8 = @as(*const [8]f32, @ptrCast(x[0..8])).*;
    const v1: Vec8 = @as(*const [8]f32, @ptrCast(x[8..16])).*;
    const ss = v0 * v0 + v1 * v1;
    const s: f32 = @reduce(.Add, ss);
    const ms = s / @as(f32, @floatFromInt(n));
    const sc = 1.0 / @sqrt(ms + 1e-5);
    const vs: Vec8 = @splat(sc);
    @as(*[8]f32, @ptrCast(out[0..8])).* = v0 * vs;
    @as(*[8]f32, @ptrCast(out[8..16])).* = v1 * vs;
    return sc;
}

inline fn softmax_fwd_precise(logits: []const f32, n: usize, probs: []f32) void {
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

inline fn linear_bwd_x(w: []const f32, dout: []const f32, nout: usize, nin: usize, dx: []f32) void {
    // dx += W^T · dout, rows walked sequentially so w is read linearly
    if (nin == N_EMBD) {
        var a0: Vec8 = @splat(0);
        var a1: Vec8 = @splat(0);
        for (0..nout) |r| {
            const dr: Vec8 = @splat(dout[r]);
            const row: *const [16]f32 = @ptrCast(w.ptr + r * 16);
            a0 = @mulAdd(Vec8, dr, row[0..8].*, a0);
            a1 = @mulAdd(Vec8, dr, row[8..16].*, a1);
        }
        const d0: Vec8 = @as(*const [8]f32, @ptrCast(dx[0..8])).*;
        const d1: Vec8 = @as(*const [8]f32, @ptrCast(dx[8..16])).*;
        @as(*[8]f32, @ptrCast(dx[0..8])).* = d0 + a0;
        @as(*[8]f32, @ptrCast(dx[8..16])).* = d1 + a1;
    } else if (nin == MLP_DIM) {
        var acc: [8]Vec8 = [_]Vec8{@splat(0)} ** 8;
        for (0..nout) |r| {
            const dr: Vec8 = @splat(dout[r]);
            const row: [*c]const f32 = w.ptr + r * MLP_DIM;
            inline for (0..8) |k| {
                const wv: Vec8 = @as(*const [8]f32, @ptrCast(row + k * 8)).*;
                acc[k] = @mulAdd(Vec8, dr, wv, acc[k]);
            }
        }
        inline for (0..8) |k| {
            const d: Vec8 = @as(*const [8]f32, @ptrCast(dx[k * 8 ..][0..8])).*;
            @as(*[8]f32, @ptrCast(dx[k * 8 ..][0..8])).* = d + acc[k];
        }
    } else {
        for (0..nin) |c_idx| {
            var s: f32 = 0;
            for (0..nout) |r| s += dout[r] * w[r * nin + c_idx];
            dx[c_idx] += s;
        }
    }
}

inline fn linear_bwd_w(x: []const f32, dout: []const f32, nout: usize, nin: usize, dw: []f32) void {
    // dw += dout ⊗ x, one 8-wide FMA per weight vector
    if (nin == N_EMBD) {
        const xv0: Vec8 = @as(*const [8]f32, @ptrCast(x[0..8])).*;
        const xv1: Vec8 = @as(*const [8]f32, @ptrCast(x[8..16])).*;
        for (0..nout) |r| {
            const dr: Vec8 = @splat(dout[r]);
            const dwp: [*c]f32 = dw.ptr + r * 16;
            const c0: Vec8 = @as(*const [8]f32, @ptrCast(dwp)).*;
            const c1: Vec8 = @as(*const [8]f32, @ptrCast(dwp + 8)).*;
            @as(*[8]f32, @ptrCast(dwp)).* = @mulAdd(Vec8, dr, xv0, c0);
            @as(*[8]f32, @ptrCast(dwp + 8)).* = @mulAdd(Vec8, dr, xv1, c1);
        }
    } else if (nin == MLP_DIM) {
        var xv: [8]Vec8 = undefined;
        inline for (0..8) |k| {
            xv[k] = @as(*const [8]f32, @ptrCast(x[k * 8 ..][0..8])).*;
        }
        for (0..nout) |r| {
            const dr: Vec8 = @splat(dout[r]);
            const dwp: [*c]f32 = dw.ptr + r * MLP_DIM;
            inline for (0..8) |k| {
                const c: Vec8 = @as(*const [8]f32, @ptrCast(dwp + k * 8)).*;
                @as(*[8]f32, @ptrCast(dwp + k * 8)).* = @mulAdd(Vec8, dr, xv[k], c);
            }
        }
    } else {
        for (0..nout) |r| {
            const dr = dout[r];
            const dwr = dw[r * nin ..][0..nin];
            for (0..nin) |c_idx| dwr[c_idx] += dr * x[c_idx];
        }
    }
}

inline fn rmsnorm_bwd(x: []const f32, scale: f32, dout: []const f32, n: usize, dx: []f32) void {
    if (n == N_EMBD) {
        const v0: Vec8 = @as(*const [8]f32, @ptrCast(x[0..8])).*;
        const v1: Vec8 = @as(*const [8]f32, @ptrCast(x[8..16])).*;
        const g0: Vec8 = @as(*const [8]f32, @ptrCast(dout[0..8])).*;
        const g1: Vec8 = @as(*const [8]f32, @ptrCast(dout[8..16])).*;
        const dot = @reduce(.Add, @mulAdd(Vec8, g0, v0, g1 * v1));
        const coeff = scale * scale * scale / @as(f32, @floatFromInt(n));
        const vs: Vec8 = @splat(scale);
        const cs: Vec8 = @splat(coeff * dot);
        const x0: Vec8 = @as(*const [8]f32, @ptrCast(dx[0..8])).*;
        const x1: Vec8 = @as(*const [8]f32, @ptrCast(dx[8..16])).*;
        @as(*[8]f32, @ptrCast(dx[0..8])).* = @mulAdd(Vec8, g0, vs, x0 - v0 * cs);
        @as(*[8]f32, @ptrCast(dx[8..16])).* = @mulAdd(Vec8, g1, vs, x1 - v1 * cs);
    } else {
        var dot: f32 = 0;
        for (0..n) |i| dot += dout[i] * x[i];
        const coeff = scale * scale * scale / @as(f32, @floatFromInt(n));
        for (0..n) |i| dx[i] += scale * dout[i] - coeff * x[i] * dot;
    }
}

fn gpt_forward(token_id: usize, pos_id: usize, logits_out: []f32, act: *PosActs) void {
    var x: [N_EMBD]f32 = undefined;
    var tmp: [MLP_DIM]f32 = undefined;
    {
        const e0: Vec8 = @as(*const [8]f32, @ptrCast(wte[token_id * N_EMBD ..][0..8])).*;
        const e1: Vec8 = @as(*const [8]f32, @ptrCast(wte[token_id * N_EMBD + 8 ..][0..8])).*;
        const p0: Vec8 = @as(*const [8]f32, @ptrCast(wpe[pos_id * N_EMBD ..][0..8])).*;
        const p1: Vec8 = @as(*const [8]f32, @ptrCast(wpe[pos_id * N_EMBD + 8 ..][0..8])).*;
        @as(*[8]f32, @ptrCast(x[0..8])).* = e0 + p0;
        @as(*[8]f32, @ptrCast(x[8..16])).* = e1 + p1;
    }
    @memcpy(&act.x_embed, &x);
    act.rms_scale_init = rmsnorm_fwd(&x, N_EMBD, &x);

    for (0..N_LAYER) |li| {
        @memcpy(&act.x_in[li], &x);
        var xn: [N_EMBD]f32 = undefined;
        act.rms_scale_attn[li] = rmsnorm_fwd(&x, N_EMBD, &xn);
        // quantize the normed value before it feeds q/k/v (per-token 8 bit);
        // the saved copy must match, since backward reads what forward ate
        AQ(&xn);
        @memcpy(&act.xn_attn[li], &xn);

        var q: [N_EMBD]f32 = undefined;
        var k: [N_EMBD]f32 = undefined;
        var v: [N_EMBD]f32 = undefined;
        linear_fwd(&xn, TW(attn_wq[li], qw_q[li][0..]), N_EMBD, N_EMBD, &q);
        linear_fwd(&xn, TW(attn_wk[li], qw_k[li][0..]), N_EMBD, N_EMBD, &k);
        linear_fwd(&xn, TW(attn_wv[li], qw_v[li][0..]), N_EMBD, N_EMBD, &v);
        @memcpy(&act.q[li], &q);
        @memcpy(&kv_keys[li][pos_id], &k);
        @memcpy(&kv_vals[li][pos_id], &v);
        const seq_len = pos_id + 1;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(N_EMBD / N_HEAD)));

        var ao: [N_EMBD]f32 = [_]f32{0} ** N_EMBD;
        for (0..N_HEAD) |h| {
            const hs = h * HEAD_DIM;
            var al: [BLOCK_SIZE]f32 = undefined;
            for (0..seq_len) |tt| {
                const qh: *const [4]f32 = @ptrCast(&q[hs]);
                const kh: *const [4]f32 = @ptrCast(&kv_keys[li][tt][hs]);
                al[tt] = dot4(qh, kh) * scale;
            }
            var mx = al[0];
            for (1..seq_len) |tt| {
                if (al[tt] > mx) mx = al[tt];
            }
            var sm: f32 = 0;
            for (0..seq_len) |tt| {
                al[tt] = fexpf(al[tt] - mx);
                sm += al[tt];
            }
            const inv: f32 = 1.0 / sm;
            for (0..seq_len) |tt| al[tt] *= inv;
            for (0..seq_len) |tt| act.aw[li][h][tt] = al[tt];
            var vacc: Vec4 = @splat(0);
            for (0..seq_len) |tt| {
                const av: Vec4 = @splat(al[tt]);
                const vt: Vec4 = @as(*const [4]f32, @ptrCast(&kv_vals[li][tt][hs])).*;
                vacc = @mulAdd(Vec4, av, vt, vacc);
            }
            @as(*[4]f32, @ptrCast(ao[hs..][0..4])).* = vacc;
        }
        AQ(&ao);
        @memcpy(&act.attn_out[li], &ao);
        linear_fwd(&ao, TW(attn_wo[li], qw_o[li][0..]), N_EMBD, N_EMBD, tmp[0..N_EMBD]);
        {
            const t0: Vec8 = @as(*const [8]f32, @ptrCast(tmp[0..8])).*;
            const t1: Vec8 = @as(*const [8]f32, @ptrCast(tmp[8..16])).*;
            const r0: Vec8 = @as(*const [8]f32, @ptrCast(&act.x_in[li][0])).*;
            const r1: Vec8 = @as(*const [8]f32, @ptrCast(&act.x_in[li][8])).*;
            @as(*[8]f32, @ptrCast(x[0..8])).* = t0 + r0;
            @as(*[8]f32, @ptrCast(x[8..16])).* = t1 + r1;
        }
        @memcpy(&act.x_mid[li], &x);

        var xn_m: [N_EMBD]f32 = undefined;
        act.rms_scale_mlp[li] = rmsnorm_fwd(&x, N_EMBD, &xn_m);
        AQ(&xn_m);
        @memcpy(&act.xn_mlp[li], &xn_m);

        var h1: [MLP_DIM]f32 = undefined;
        linear_fwd(&xn_m, TW(mlp_fc1[li], qw_fc1[li][0..]), MLP_DIM, N_EMBD, &h1);
        @memcpy(&act.mlp_pre[li], &h1);
        var h2: [MLP_DIM]f32 = undefined;
        {
            const z: Vec8 = @splat(0);
            inline for (0..MLP_DIM / 8) |kk| {
                const hv: Vec8 = @as(*const [8]f32, @ptrCast(h1[kk * 8 ..][0..8])).*;
                const r = @max(hv, z);
                @as(*[8]f32, @ptrCast(h2[kk * 8 ..][0..8])).* = r * r;
            }
        }
        AQ(&h2);
        @memcpy(&act.mlp_post[li], &h2);
        linear_fwd(@ptrCast(@as(*const [MLP_DIM]f32, &h2)), TW(mlp_fc2[li], qw_fc2[li][0..]), N_EMBD, MLP_DIM, tmp[0..N_EMBD]);
        {
            const t0: Vec8 = @as(*const [8]f32, @ptrCast(tmp[0..8])).*;
            const t1: Vec8 = @as(*const [8]f32, @ptrCast(tmp[8..16])).*;
            const m0: Vec8 = @as(*const [8]f32, @ptrCast(&act.x_mid[li][0])).*;
            const m1: Vec8 = @as(*const [8]f32, @ptrCast(&act.x_mid[li][8])).*;
            @as(*[8]f32, @ptrCast(x[0..8])).* = t0 + m0;
            @as(*[8]f32, @ptrCast(x[8..16])).* = t1 + m1;
        }
    }
    AQ(&x);
    @memcpy(&act.x_out, &x);
    linear_fwd(&x, TW(lm_head, qw_lm), vocab_size, N_EMBD, logits_out);
}

fn gpt_backward(n: usize, tokens: []const i32, targets: []const i32) void {
    for (0..N_LAYER) |li| {
        for (0..n) |tt| {
            @memset(&dk_accum[li][tt], 0);
            @memset(&dv_accum[li][tt], 0);
        }
    }
    const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(n));
    var pos: usize = n;
    while (pos > 0) {
        pos -= 1;
        const act = &saved[pos];
        const seq_len = pos + 1;
        var dl: [MAX_CHARS + 1]f32 = undefined;
        for (0..vocab_size) |i| {
            const one: f32 = if (@as(i32, @intCast(i)) == targets[pos]) 1.0 else 0.0;
            dl[i] = (saved_probs[pos][i] - one) * inv_n;
        }
        var dx: [N_EMBD]f32 = [_]f32{0} ** N_EMBD;
        linear_bwd_x(TW(lm_head, qw_lm), dl[0..vocab_size], vocab_size, N_EMBD, &dx);
        linear_bwd_w(&act.x_out, dl[0..vocab_size], vocab_size, N_EMBD, d_lm_head);

        var li: usize = N_LAYER;
        while (li > 0) {
            li -= 1;
            var d_h2: [MLP_DIM]f32 = [_]f32{0} ** MLP_DIM;
            linear_bwd_x(TW(mlp_fc2[li], qw_fc2[li][0..]), &dx, N_EMBD, MLP_DIM, &d_h2);
            linear_bwd_w(&act.mlp_post[li], &dx, N_EMBD, MLP_DIM, d_mlp_fc2[li]);
            var d_h1: [MLP_DIM]f32 = undefined;
            for (0..MLP_DIM) |i| d_h1[i] = if (act.mlp_pre[li][i] > 0) 2.0 * act.mlp_pre[li][i] * d_h2[i] else 0;
            var d_xn_mlp: [N_EMBD]f32 = [_]f32{0} ** N_EMBD;
            linear_bwd_x(TW(mlp_fc1[li], qw_fc1[li][0..]), &d_h1, MLP_DIM, N_EMBD, &d_xn_mlp);
            linear_bwd_w(&act.xn_mlp[li], &d_h1, MLP_DIM, N_EMBD, d_mlp_fc1[li]);
            var d_x_mid: [N_EMBD]f32 = [_]f32{0} ** N_EMBD;
            rmsnorm_bwd(&act.x_mid[li], act.rms_scale_mlp[li], &d_xn_mlp, N_EMBD, &d_x_mid);
            {
                const a0: Vec8 = @as(*const [8]f32, @ptrCast(dx[0..8])).*;
                const a1: Vec8 = @as(*const [8]f32, @ptrCast(dx[8..16])).*;
                const b0: Vec8 = @as(*const [8]f32, @ptrCast(&d_x_mid[0])).*;
                const b1: Vec8 = @as(*const [8]f32, @ptrCast(&d_x_mid[8])).*;
                @as(*[8]f32, @ptrCast(dx[0..8])).* = a0 + b0;
                @as(*[8]f32, @ptrCast(dx[8..16])).* = a1 + b1;
            }

            var d_ao: [N_EMBD]f32 = [_]f32{0} ** N_EMBD;
            linear_bwd_x(TW(attn_wo[li], qw_o[li][0..]), &dx, N_EMBD, N_EMBD, &d_ao);
            linear_bwd_w(&act.attn_out[li], &dx, N_EMBD, N_EMBD, d_attn_wo[li]);

            var d_q: [N_EMBD]f32 = [_]f32{0} ** N_EMBD;
            const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(N_EMBD / N_HEAD)));
            const sc4: Vec4 = @splat(scale);
            for (0..N_HEAD) |h| {
                const hs = h * HEAD_DIM;
                var d_aw: [BLOCK_SIZE]f32 = [_]f32{0} ** BLOCK_SIZE;
                const dao: Vec4 = @as(*const [4]f32, @ptrCast(d_ao[hs..][0..4])).*;
                for (0..seq_len) |tt| {
                    const vt: Vec4 = @as(*const [4]f32, @ptrCast(&kv_vals[li][tt][hs])).*;
                    d_aw[tt] = @reduce(.Add, dao * vt);
                    const av: Vec4 = @splat(act.aw[li][h][tt]);
                    const dv: Vec4 = @as(*const [4]f32, @ptrCast(&dv_accum[li][tt][hs])).*;
                    @as(*[4]f32, @ptrCast(&dv_accum[li][tt][hs])).* = @mulAdd(Vec4, av, dao, dv);
                }
                var dot: f32 = 0;
                for (0..seq_len) |tt| dot += d_aw[tt] * act.aw[li][h][tt];
                var d_al: [BLOCK_SIZE]f32 = undefined;
                for (0..seq_len) |tt| d_al[tt] = act.aw[li][h][tt] * (d_aw[tt] - dot);
                const qv: Vec4 = @as(*const [4]f32, @ptrCast(&act.q[li][hs])).*;
                var dq_acc: Vec4 = @splat(0);
                for (0..seq_len) |tt| {
                    const dal: Vec4 = @splat(d_al[tt]);
                    const kt: Vec4 = @as(*const [4]f32, @ptrCast(&kv_keys[li][tt][hs])).*;
                    dq_acc = @mulAdd(Vec4, kt * dal, sc4, dq_acc);
                    const dkv: Vec4 = @as(*const [4]f32, @ptrCast(&dk_accum[li][tt][hs])).*;
                    @as(*[4]f32, @ptrCast(&dk_accum[li][tt][hs])).* = @mulAdd(Vec4, qv * dal, sc4, dkv);
                }
                @as(*[4]f32, @ptrCast(&d_q[hs])).* = dq_acc;
            }
            var d_xn: [N_EMBD]f32 = [_]f32{0} ** N_EMBD;
            linear_bwd_x(TW(attn_wq[li], qw_q[li][0..]), &d_q, N_EMBD, N_EMBD, &d_xn);
            linear_bwd_w(&act.xn_attn[li], &d_q, N_EMBD, N_EMBD, d_attn_wq[li]);
            linear_bwd_x(TW(attn_wk[li], qw_k[li][0..]), &dk_accum[li][pos], N_EMBD, N_EMBD, &d_xn);
            linear_bwd_w(&act.xn_attn[li], &dk_accum[li][pos], N_EMBD, N_EMBD, d_attn_wk[li]);
            linear_bwd_x(TW(attn_wv[li], qw_v[li][0..]), &dv_accum[li][pos], N_EMBD, N_EMBD, &d_xn);
            linear_bwd_w(&act.xn_attn[li], &dv_accum[li][pos], N_EMBD, N_EMBD, d_attn_wv[li]);

            var d_x_in: [N_EMBD]f32 = [_]f32{0} ** N_EMBD;
            rmsnorm_bwd(&act.x_in[li], act.rms_scale_attn[li], &d_xn, N_EMBD, &d_x_in);
            {
                const a0: Vec8 = @as(*const [8]f32, @ptrCast(dx[0..8])).*;
                const a1: Vec8 = @as(*const [8]f32, @ptrCast(dx[8..16])).*;
                const b0: Vec8 = @as(*const [8]f32, @ptrCast(&d_x_in[0])).*;
                const b1: Vec8 = @as(*const [8]f32, @ptrCast(&d_x_in[8])).*;
                @as(*[8]f32, @ptrCast(dx[0..8])).* = a0 + b0;
                @as(*[8]f32, @ptrCast(dx[8..16])).* = a1 + b1;
            }
        }
        var d_embed: [N_EMBD]f32 = [_]f32{0} ** N_EMBD;
        rmsnorm_bwd(&act.x_embed, act.rms_scale_init, &dx, N_EMBD, &d_embed);
        const tok: usize = @intCast(tokens[pos]);
        {
            const de0: Vec8 = @as(*const [8]f32, @ptrCast(&d_embed[0])).*;
            const de1: Vec8 = @as(*const [8]f32, @ptrCast(&d_embed[8])).*;
            const wtp: [*c]f32 = d_wte.ptr + tok * N_EMBD;
            const wpp: [*c]f32 = d_wpe.ptr + pos * N_EMBD;
            const w0: Vec8 = @as(*const [8]f32, @ptrCast(wtp)).*;
            const w1: Vec8 = @as(*const [8]f32, @ptrCast(wtp + 8)).*;
            @as(*[8]f32, @ptrCast(wtp)).* = w0 + de0;
            @as(*[8]f32, @ptrCast(wtp + 8)).* = w1 + de1;
            const p0: Vec8 = @as(*const [8]f32, @ptrCast(wpp)).*;
            const p1: Vec8 = @as(*const [8]f32, @ptrCast(wpp + 8)).*;
            @as(*[8]f32, @ptrCast(wpp)).* = p0 + de0;
            @as(*[8]f32, @ptrCast(wpp + 8)).* = p1 + de1;
        }
    }
}

// ---------------------------------------------------------------------------
// optimizer
// ---------------------------------------------------------------------------

fn adam_update(p: []f32, g: []f32, m: []f32, v: []f32, lr: f32, b1: f32, b2: f32, eps: f32, b1c: f32, b2c: f32, gscale: f32) void {
    const vb1: Vec8 = @splat(b1);
    const vc1: Vec8 = @splat(1 - b1);
    const vb2: Vec8 = @splat(b2);
    const vc2: Vec8 = @splat(1 - b2);
    const vb1c: Vec8 = @splat(b1c);
    const vb2c: Vec8 = @splat(b2c);
    const veps: Vec8 = @splat(eps);
    const vlr: Vec8 = @splat(lr);
    const vgs: Vec8 = @splat(gscale);
    const n = p.len;
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const gv: Vec8 = @as(*const [8]f32, @ptrCast(g[i..][0..8])).* * vgs;
        var mv: Vec8 = @as(*const [8]f32, @ptrCast(m[i..][0..8])).*;
        var vv: Vec8 = @as(*const [8]f32, @ptrCast(v[i..][0..8])).*;
        mv = mv * vb1 + gv * vc1;
        vv = vv * vb2 + gv * gv * vc2;
        @as(*[8]f32, @ptrCast(m[i..][0..8])).* = mv;
        @as(*[8]f32, @ptrCast(v[i..][0..8])).* = vv;
        const upd = (vlr * (mv / vb1c)) / (@sqrt(vv / vb2c) + veps);
        const pv: Vec8 = @as(*const [8]f32, @ptrCast(p[i..][0..8])).*;
        @as(*[8]f32, @ptrCast(p[i..][0..8])).* = pv - upd;
        @as(*[8]f32, @ptrCast(g[i..][0..8])).* = @splat(0);
    }
    while (i < n) : (i += 1) {
        const gs = g[i] * gscale;
        m[i] = b1 * m[i] + (1 - b1) * gs;
        v[i] = b2 * v[i] + (1 - b2) * gs * gs;
        p[i] -= lr * (m[i] / b1c) / (@sqrt(v[i] / b2c) + eps);
        g[i] = 0;
    }
}

fn reset_adam() void {
    const es = vocab_size * N_EMBD;
    const ps = BLOCK_SIZE * N_EMBD;
    const as = N_EMBD * N_EMBD;
    const ms = MLP_DIM * N_EMBD;
    @memset(adam_m_wte[0..es], 0);
    @memset(adam_v_wte[0..es], 0);
    @memset(adam_m_wpe[0..ps], 0);
    @memset(adam_v_wpe[0..ps], 0);
    @memset(adam_m_lm[0..es], 0);
    @memset(adam_v_lm[0..es], 0);
    for (0..N_LAYER) |l| {
        @memset(adam_m_wq[l][0..as], 0);
        @memset(adam_v_wq[l][0..as], 0);
        @memset(adam_m_wk[l][0..as], 0);
        @memset(adam_v_wk[l][0..as], 0);
        @memset(adam_m_wv[l][0..as], 0);
        @memset(adam_v_wv[l][0..as], 0);
        @memset(adam_m_wo[l][0..as], 0);
        @memset(adam_v_wo[l][0..as], 0);
        @memset(adam_m_fc1[l][0..ms], 0);
        @memset(adam_v_fc1[l][0..ms], 0);
        @memset(adam_m_fc2[l][0..ms], 0);
        @memset(adam_v_fc2[l][0..ms], 0);
    }
}

// ---------------------------------------------------------------------------
// fast packed inference path (from microgpt.zig, unchanged math)
// ---------------------------------------------------------------------------

var iw_q: [N_LAYER][N_EMBD * N_EMBD]f32 align(64) = [_][N_EMBD * N_EMBD]f32{[_]f32{0} ** (N_EMBD * N_EMBD)} ** N_LAYER;
var iw_k: [N_LAYER][N_EMBD * N_EMBD]f32 align(64) = [_][N_EMBD * N_EMBD]f32{[_]f32{0} ** (N_EMBD * N_EMBD)} ** N_LAYER;
var iw_v: [N_LAYER][N_EMBD * N_EMBD]f32 align(64) = [_][N_EMBD * N_EMBD]f32{[_]f32{0} ** (N_EMBD * N_EMBD)} ** N_LAYER;
var iw_o: [N_LAYER][N_EMBD * N_EMBD]f32 align(64) = [_][N_EMBD * N_EMBD]f32{[_]f32{0} ** (N_EMBD * N_EMBD)} ** N_LAYER;
var iw_fc1: [N_LAYER][N_EMBD * MLP_DIM]f32 align(64) = [_][N_EMBD * MLP_DIM]f32{[_]f32{0} ** (N_EMBD * MLP_DIM)} ** N_LAYER;
var iw_fc2: [N_LAYER][MLP_DIM * N_EMBD]f32 align(64) = [_][MLP_DIM * N_EMBD]f32{[_]f32{0} ** (MLP_DIM * N_EMBD)} ** N_LAYER;
var iw_lm: [N_EMBD * LM_PAD_MAX]f32 align(64) = [_]f32{0} ** (N_EMBD * LM_PAD_MAX);

fn pack_t(src: []const f32, dst: []f32, nout: usize, nin: usize) void {
    for (0..nout) |r| {
        for (0..nin) |c_idx| {
            dst[c_idx * nout + r] = src[r * nin + c_idx];
        }
    }
}

const PreTok = struct {
    xin: [N_EMBD]f32,
    q: [N_EMBD]f32,
    k: [N_EMBD]f32,
    v: [N_EMBD]f32,
};

var pretok: []PreTok = undefined;

inline fn vfexpq4(x: Vec4) Vec4 {
    const y: Vec4 = x * @as(Vec4, @splat(12102203.1615614 * 1.4426950408));
    const yi: @Vector(4, i32) = @intFromFloat(y);
    const bi: @Vector(4, i32) = yi + @as(@Vector(4, i32), @splat(1065353216));
    return @bitCast(bi);
}

inline fn hsum4(v: Vec4) f32 {
    return @reduce(.Add, v);
}

fn sample_logits(logits: []f32, n: usize, npad: usize, inv_t: f32) usize {
    var p: [LM_PAD_MAX]f32 = undefined;
    for (n..npad) |i| logits[i] = logits[0];

    var m = logits[0];
    for (1..npad) |i| {
        if (logits[i] > m) m = logits[i];
    }
    const vmx: f32 = m * inv_t;

    var s0: Vec4 = @splat(0);
    var s1: Vec4 = @splat(0);
    var i: usize = 0;
    while (i + 8 <= npad) : (i += 8) {
        const l0: Vec4 = logits[i..][0..4].*;
        const l1: Vec4 = logits[i + 4 ..][0..4].*;
        const e0 = vfexpq4(l0 * @as(Vec4, @splat(inv_t)) - @as(Vec4, @splat(vmx)));
        const e1 = vfexpq4(l1 * @as(Vec4, @splat(inv_t)) - @as(Vec4, @splat(vmx)));
        p[i..][0..4].* = e0;
        p[i + 4 ..][0..4].* = e1;
        s0 += e0;
        s1 += e1;
    }
    while (i < npad) : (i += 4) {
        const l0: Vec4 = logits[i..][0..4].*;
        const e0 = vfexpq4(l0 * @as(Vec4, @splat(inv_t)) - @as(Vec4, @splat(vmx)));
        p[i..][0..4].* = e0;
        s0 += e0;
    }
    const sum: f32 = hsum4(s0 + s1) - @as(f32, @floatFromInt(npad - n)) * p[0];
    const r: f32 = @as(f32, @floatCast(rng_uniform())) * sum;
    var cum: f32 = 0;
    for (0..n) |k| {
        cum += p[k];
        if (r < cum) return k;
    }
    return n - 1;
}

const Vec16 = struct { lo: Vec8, hi: Vec8 };

inline fn mv16_blk16r(x: *const [N_EMBD]f32, wcol: [*c]const f32, ldw: usize) Vec16 {
    var a0: Vec8 = @splat(0);
    var a1: Vec8 = @splat(0);
    var b0: Vec8 = @splat(0);
    var b1: Vec8 = @splat(0);
    var c_idx: usize = 0;
    while (c_idx < 16) : (c_idx += 2) {
        const bc0: Vec8 = @splat(x[c_idx]);
        const w0_lo: Vec8 = @as(*const [8]f32, @ptrCast(wcol + c_idx * ldw)).*;
        const w0_hi: Vec8 = @as(*const [8]f32, @ptrCast(wcol + c_idx * ldw + 8)).*;
        a0 = @mulAdd(Vec8, w0_lo, bc0, a0);
        a1 = @mulAdd(Vec8, w0_hi, bc0, a1);
        const bc1: Vec8 = @splat(x[c_idx + 1]);
        const w1_lo: Vec8 = @as(*const [8]f32, @ptrCast(wcol + (c_idx + 1) * ldw)).*;
        const w1_hi: Vec8 = @as(*const [8]f32, @ptrCast(wcol + (c_idx + 1) * ldw + 8)).*;
        b0 = @mulAdd(Vec8, w1_lo, bc1, b0);
        b1 = @mulAdd(Vec8, w1_hi, bc1, b1);
    }
    return .{ .lo = a0 + b0, .hi = a1 + b1 };
}

inline fn mv16_blk16r4(x: *const [N_EMBD]f32, wcol: [*c]const f32, ldw: usize) Vec16 {
    var a0: Vec8 = @splat(0);
    var a1: Vec8 = @splat(0);
    var b0: Vec8 = @splat(0);
    var b1: Vec8 = @splat(0);
    var c0v: Vec8 = @splat(0);
    var c1v: Vec8 = @splat(0);
    var d0v: Vec8 = @splat(0);
    var d1v: Vec8 = @splat(0);
    var c_idx: usize = 0;
    while (c_idx < 16) : (c_idx += 4) {
        const bc0: Vec8 = @splat(x[c_idx]);
        a0 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + c_idx * ldw)).*, bc0, a0);
        a1 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + c_idx * ldw + 8)).*, bc0, a1);
        const bc1: Vec8 = @splat(x[c_idx + 1]);
        b0 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + (c_idx + 1) * ldw)).*, bc1, b0);
        b1 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + (c_idx + 1) * ldw + 8)).*, bc1, b1);
        const bc2: Vec8 = @splat(x[c_idx + 2]);
        c0v = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + (c_idx + 2) * ldw)).*, bc2, c0v);
        c1v = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + (c_idx + 2) * ldw + 8)).*, bc2, c1v);
        const bc3: Vec8 = @splat(x[c_idx + 3]);
        d0v = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + (c_idx + 3) * ldw)).*, bc3, d0v);
        d1v = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + (c_idx + 3) * ldw + 8)).*, bc3, d1v);
    }
    return .{ .lo = (a0 + b0) + (c0v + d0v), .hi = (a1 + b1) + (c1v + d1v) };
}

inline fn mv16_blk16(x: *const [N_EMBD]f32, wcol: [*c]const f32, ldw: usize, out: *[N_EMBD]f32) void {
    const r = mv16_blk16r(x, wcol, ldw);
    @as(*[8]f32, @ptrCast(out[0..8])).* = r.lo;
    @as(*[8]f32, @ptrCast(out[8..16])).* = r.hi;
}

inline fn mv16_blk16v(xr: Vec16, wcol: [*c]const f32, ldw: usize) Vec16 {
    var a0: Vec8 = @splat(0);
    var a1: Vec8 = @splat(0);
    var b0: Vec8 = @splat(0);
    var b1: Vec8 = @splat(0);
    const lo_arr: [8]f32 = xr.lo;
    const hi_arr: [8]f32 = xr.hi;
    var c_idx: usize = 0;
    while (c_idx < 8) : (c_idx += 2) {
        const bc0: Vec8 = @splat(lo_arr[c_idx]);
        a0 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + c_idx * ldw)).*, bc0, a0);
        a1 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + c_idx * ldw + 8)).*, bc0, a1);
        const bc1: Vec8 = @splat(lo_arr[c_idx + 1]);
        b0 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + (c_idx + 1) * ldw)).*, bc1, b0);
        b1 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + (c_idx + 1) * ldw + 8)).*, bc1, b1);
    }
    var c2: usize = 0;
    while (c2 < 8) : (c2 += 2) {
        const bc0: Vec8 = @splat(hi_arr[c2]);
        a0 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + (8 + c2) * ldw)).*, bc0, a0);
        a1 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + (8 + c2) * ldw + 8)).*, bc0, a1);
        const bc1: Vec8 = @splat(hi_arr[c2 + 1]);
        b0 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + (8 + c2 + 1) * ldw)).*, bc1, b0);
        b1 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + (8 + c2 + 1) * ldw + 8)).*, bc1, b1);
    }
    return .{ .lo = a0 + b0, .hi = a1 + b1 };
}

inline fn mv16_blk8(x: *const [N_EMBD]f32, wcol: [*c]const f32, ldw: usize, out: *[8]f32) void {
    var a: Vec8 = @splat(0);
    var b: Vec8 = @splat(0);
    var c_idx: usize = 0;
    while (c_idx < 16) : (c_idx += 2) {
        a = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + c_idx * ldw)).*, @as(Vec8, @splat(x[c_idx])), a);
        b = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(wcol + (c_idx + 1) * ldw)).*, @as(Vec8, @splat(x[c_idx + 1])), b);
    }
    @as(*[8]f32, @ptrCast(out)).* = a + b;
}

inline fn mv16_blk4(x: *const [N_EMBD]f32, wcol: [*c]const f32, ldw: usize, out: *[4]f32) void {
    var a: Vec4 = @splat(0);
    var b: Vec4 = @splat(0);
    var c_idx: usize = 0;
    while (c_idx < 16) : (c_idx += 2) {
        a = @mulAdd(Vec4, @as(*const [4]f32, @ptrCast(wcol + c_idx * ldw)).*, @as(Vec4, @splat(x[c_idx])), a);
        b = @mulAdd(Vec4, @as(*const [4]f32, @ptrCast(wcol + (c_idx + 1) * ldw)).*, @as(Vec4, @splat(x[c_idx + 1])), b);
    }
    out.* = a + b;
}

inline fn rmsnorm_scale(x: *const [N_EMBD]f32) f32 {
    const v0: Vec8 = @as(*const [8]f32, @ptrCast(x[0..8])).*;
    const v1: Vec8 = @as(*const [8]f32, @ptrCast(x[8..16])).*;
    const ss = v0 * v0 + v1 * v1;
    const s: f32 = @reduce(.Add, ss);
    return 1.0 / @sqrt(s * (1.0 / @as(f32, @floatFromInt(N_EMBD))) + 1e-5);
}

inline fn rmsnorm_infer(x: *const [N_EMBD]f32, out: *[N_EMBD]f32) void {
    const v0: Vec8 = @as(*const [8]f32, @ptrCast(x[0..8])).*;
    const v1: Vec8 = @as(*const [8]f32, @ptrCast(x[8..16])).*;
    const ss = v0 * v0 + v1 * v1;
    const ms: f32 = @reduce(.Add, ss) * (1.0 / @as(f32, @floatFromInt(N_EMBD)));
    const vs: Vec8 = @splat(1.0 / @sqrt(ms + 1e-5));
    @as(*[8]f32, @ptrCast(out[0..8])).* = v0 * vs;
    @as(*[8]f32, @ptrCast(out[8..16])).* = v1 * vs;
}

fn build_pretok() void {
    const n = vocab_size * BLOCK_SIZE;
    const bytes = (n * @sizeOf(PreTok) + 63) & ~@as(usize, 63);
    const mem = gpa.alignedAlloc(u8, .fromByteUnits(64), bytes) catch {
        std.debug.print("out of memory\n", .{});
        clib.exit(1);
        unreachable;
    };
    pretok = @as([*]PreTok, @ptrCast(@alignCast(mem.ptr)))[0..n];
    for (0..vocab_size) |t| {
        for (0..BLOCK_SIZE) |p| {
            var x: [N_EMBD]f32 = undefined;
            for (0..N_EMBD) |i| x[i] = wte[t * N_EMBD + i] + wpe[p * N_EMBD + i];
            var xr: [N_EMBD]f32 = undefined;
            rmsnorm_infer(&x, &xr);
            const e = &pretok[t * BLOCK_SIZE + p];
            @memcpy(&e.xin, &xr);
            var xn: [N_EMBD]f32 = undefined;
            rmsnorm_infer(&xr, &xn);
            mv16_blk16(&xn, &iw_q[0][0], N_EMBD, &e.q);
            mv16_blk16(&xn, &iw_k[0][0], N_EMBD, &e.k);
            mv16_blk16(&xn, &iw_v[0][0], N_EMBD, &e.v);
        }
    }
}

fn infer_pack_weights(q: bool) void {
    lm_pad = ((vocab_size + 3) / 4) * 4;
    for (0..N_LAYER) |li| {
        pack_t(if (q) qw_q[li][0..] else attn_wq[li], iw_q[li][0..], N_EMBD, N_EMBD);
        pack_t(if (q) qw_k[li][0..] else attn_wk[li], iw_k[li][0..], N_EMBD, N_EMBD);
        pack_t(if (q) qw_v[li][0..] else attn_wv[li], iw_v[li][0..], N_EMBD, N_EMBD);
        pack_t(if (q) qw_o[li][0..] else attn_wo[li], iw_o[li][0..], N_EMBD, N_EMBD);
        const src_fc1: []const f32 = if (q) qw_fc1[li][0..] else mlp_fc1[li];
        for (0..MLP_DIM / 16) |bi| {
            for (0..N_EMBD) |c_idx| {
                for (0..16) |r| {
                    iw_fc1[li][bi * 256 + c_idx * 16 + r] = src_fc1[(bi * 16 + r) * N_EMBD + c_idx];
                }
            }
        }
        pack_t(if (q) qw_fc2[li][0..] else mlp_fc2[li], iw_fc2[li][0..], N_EMBD, MLP_DIM);
    }
    @memset(iw_lm[0..], 0);
    const src_lm: []const f32 = if (q) qw_lm else lm_head;
    for (0..vocab_size) |r| {
        for (0..N_EMBD) |c_idx| {
            iw_lm[c_idx * lm_pad + r] = src_lm[r * N_EMBD + c_idx];
        }
    }
    build_pretok();
}

fn gpt_forward_infer(token_id: usize, pos_id: usize, logits_out: []f32) void {
    var x: [N_EMBD]f32 = undefined;
    var xn: [N_EMBD]f32 = undefined;
    var xin: [N_EMBD]f32 = undefined;
    var q: [N_EMBD]f32 = undefined;

    for (0..N_LAYER) |li| {
        var xin_p: *const [N_EMBD]f32 = undefined;
        if (li == 0) {
            const pt = &pretok[token_id * BLOCK_SIZE + pos_id];
            xin_p = &pt.xin;
            @memcpy(&q, &pt.q);
            @memcpy(&kv_keys[li][pos_id], &pt.k);
            @memcpy(&kv_vals[li][pos_id], &pt.v);
        } else {
            @memcpy(&xin, &x);
            xin_p = &xin;
            rmsnorm_infer(&x, &xn);
            mv16_blk16(&xn, &iw_q[li][0], N_EMBD, &q);
            mv16_blk16(&xn, &iw_k[li][0], N_EMBD, kv_keys[li][pos_id][0..]);
            mv16_blk16(&xn, &iw_v[li][0], N_EMBD, kv_vals[li][pos_id][0..]);
        }

        const seq_len = pos_id + 1;
        const q0: Vec4 = @as(*const [4]f32, @ptrCast(q[0..4])).*;
        const q1: Vec4 = @as(*const [4]f32, @ptrCast(q[4..8])).*;
        const q2: Vec4 = @as(*const [4]f32, @ptrCast(q[8..12])).*;
        const q3: Vec4 = @as(*const [4]f32, @ptrCast(q[12..16])).*;
        var sc: [BLOCK_SIZE]Vec4 = undefined;

        var m0: Vec4 = @splat(-3.0e38);
        var m1: Vec4 = @splat(-3.0e38);
        var tt: usize = 0;
        while (tt + 1 < seq_len) : (tt += 2) {
            {
                const kt = &kv_keys[li][tt];
                const y0 = q0 * @as(Vec4, kt[0..4].*);
                const y1 = q1 * @as(Vec4, kt[4..8].*);
                const y2 = q2 * @as(Vec4, kt[8..12].*);
                const y3 = q3 * @as(Vec4, kt[12..16].*);
                const s0 = (y0 + y1) + (y2 + y3);
                sc[tt] = s0 * @as(Vec4, @splat(ATTN_SCALE));
                m0 = @max(m0, sc[tt]);
            }
            {
                const kt = &kv_keys[li][tt + 1];
                const y0 = q0 * @as(Vec4, kt[0..4].*);
                const y1 = q1 * @as(Vec4, kt[4..8].*);
                const y2 = q2 * @as(Vec4, kt[8..12].*);
                const y3 = q3 * @as(Vec4, kt[12..16].*);
                const s0 = (y0 + y1) + (y2 + y3);
                sc[tt + 1] = s0 * @as(Vec4, @splat(ATTN_SCALE));
                m1 = @max(m1, sc[tt + 1]);
            }
        }
        if (tt < seq_len) {
            const kt = &kv_keys[li][tt];
            const y0 = q0 * @as(Vec4, kt[0..4].*);
            const y1 = q1 * @as(Vec4, kt[4..8].*);
            const y2 = q2 * @as(Vec4, kt[8..12].*);
            const y3 = q3 * @as(Vec4, kt[12..16].*);
            const s0 = (y0 + y1) + (y2 + y3);
            sc[tt] = s0 * @as(Vec4, @splat(ATTN_SCALE));
            m0 = @max(m0, sc[tt]);
        }
        const mx: Vec4 = @max(m0, m1);

        var s0v: Vec4 = @splat(0);
        var s1v: Vec4 = @splat(0);
        var o0: Vec4 = @splat(0);
        var o1: Vec4 = @splat(0);
        var o2: Vec4 = @splat(0);
        var o3: Vec4 = @splat(0);
        var uu0: Vec4 = @splat(0);
        var uu1: Vec4 = @splat(0);
        var uu2: Vec4 = @splat(0);
        var uu3: Vec4 = @splat(0);

        tt = 0;
        while (tt + 1 < seq_len) : (tt += 2) {
            const e0 = vfexpq4(sc[tt] - mx);
            const e1 = vfexpq4(sc[tt + 1] - mx);
            s0v += e0;
            s1v += e1;
            {
                const vt = &kv_vals[li][tt];
                const v0: Vec4 = vt[0..4].*;
                const v1: Vec4 = vt[4..8].*;
                const v2: Vec4 = vt[8..12].*;
                const v3: Vec4 = vt[12..16].*;
                o0 += v0 * @as(Vec4, @splat(e0[0]));
                o1 += v1 * @as(Vec4, @splat(e0[1]));
                o2 += v2 * @as(Vec4, @splat(e0[2]));
                o3 += v3 * @as(Vec4, @splat(e0[3]));
            }
            {
                const vt = &kv_vals[li][tt + 1];
                const v0: Vec4 = vt[0..4].*;
                const v1: Vec4 = vt[4..8].*;
                const v2: Vec4 = vt[8..12].*;
                const v3: Vec4 = vt[12..16].*;
                uu0 += v0 * @as(Vec4, @splat(e1[0]));
                uu1 += v1 * @as(Vec4, @splat(e1[1]));
                uu2 += v2 * @as(Vec4, @splat(e1[2]));
                uu3 += v3 * @as(Vec4, @splat(e1[3]));
            }
        }
        if (tt < seq_len) {
            const e0 = vfexpq4(sc[tt] - mx);
            s0v += e0;
            const vt = &kv_vals[li][tt];
            const v0: Vec4 = vt[0..4].*;
            const v1: Vec4 = vt[4..8].*;
            const v2: Vec4 = vt[8..12].*;
            const v3: Vec4 = vt[12..16].*;
            o0 += v0 * @as(Vec4, @splat(e0[0]));
            o1 += v1 * @as(Vec4, @splat(e0[1]));
            o2 += v2 * @as(Vec4, @splat(e0[2]));
            o3 += v3 * @as(Vec4, @splat(e0[3]));
        }

        const stot = s0v + s1v;
        const inv = @as(Vec4, @splat(1.0)) / stot;
        const ao0 = (o0 + uu0) * @as(Vec4, @splat(inv[0]));
        const ao1 = (o1 + uu1) * @as(Vec4, @splat(inv[1]));
        const ao2 = (o2 + uu2) * @as(Vec4, @splat(inv[2]));
        const ao3 = (o3 + uu3) * @as(Vec4, @splat(inv[3]));
        var ao: [N_EMBD]f32 = undefined;
        @as(*[4]f32, @ptrCast(ao[0..4])).* = ao0;
        @as(*[4]f32, @ptrCast(ao[4..8])).* = ao1;
        @as(*[4]f32, @ptrCast(ao[8..12])).* = ao2;
        @as(*[4]f32, @ptrCast(ao[12..16])).* = ao3;

        const ao_vec = Vec16{ .lo = @as(*const [8]f32, @ptrCast(ao[0..8])).*, .hi = @as(*const [8]f32, @ptrCast(ao[8..16])).* };
        const wo = mv16_blk16v(ao_vec, &iw_o[li][0], N_EMBD);
        const xin_lo: Vec8 = @as(*const [8]f32, @ptrCast(xin_p[0..8])).*;
        const xin_hi: Vec8 = @as(*const [8]f32, @ptrCast(xin_p[8..16])).*;
        @as(*[8]f32, @ptrCast(x[0..8])).* = wo.lo + xin_lo;
        @as(*[8]f32, @ptrCast(x[8..16])).* = wo.hi + xin_hi;

        const s_mlp = rmsnorm_scale(&x);
        var h2: [MLP_DIM]f32 = undefined;
        var b_idx: usize = 0;
        while (b_idx < MLP_DIM) : (b_idx += 16) {
            const h = mv16_blk16r4(&x, &iw_fc1[li][b_idx * 16], N_EMBD);
            const z: Vec8 = @splat(0);
            const r0 = @max(h.lo, z);
            const r1 = @max(h.hi, z);
            @as(*[8]f32, @ptrCast(h2[b_idx..][0..8])).* = r0 * r0;
            @as(*[8]f32, @ptrCast(h2[b_idx + 8 ..][0..8])).* = r1 * r1;
        }
        var t0: Vec8 = @splat(0);
        var t1: Vec8 = @splat(0);
        var n0: Vec8 = @splat(0);
        var n1: Vec8 = @splat(0);
        var y0: Vec8 = @splat(0);
        var y1: Vec8 = @splat(0);
        var w0v: Vec8 = @splat(0);
        var w1v: Vec8 = @splat(0);
        var c_idx2: usize = 0;
        while (c_idx2 < MLP_DIM) : (c_idx2 += 16) {
            const h2_0: Vec4 = h2[c_idx2..][0..4].*;
            const h2_1: Vec4 = h2[c_idx2 + 4 ..][0..4].*;
            const h2_2: Vec4 = h2[c_idx2 + 8 ..][0..4].*;
            const h2_3: Vec4 = h2[c_idx2 + 12 ..][0..4].*;
            inline for (0..4) |k| {
                const hv = h2_0[k];
                const bc: Vec8 = @splat(hv);
                t0 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(&iw_fc2[li][(c_idx2 + k) * N_EMBD])).* , bc, t0);
                t1 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(&iw_fc2[li][(c_idx2 + k) * N_EMBD + 8])).* , bc, t1);
            }
            inline for (0..4) |k| {
                const hv = h2_1[k];
                const bc: Vec8 = @splat(hv);
                n0 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(&iw_fc2[li][(c_idx2 + 4 + k) * N_EMBD])).* , bc, n0);
                n1 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(&iw_fc2[li][(c_idx2 + 4 + k) * N_EMBD + 8])).* , bc, n1);
            }
            inline for (0..4) |k| {
                const hv = h2_2[k];
                const bc: Vec8 = @splat(hv);
                y0 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(&iw_fc2[li][(c_idx2 + 8 + k) * N_EMBD])).* , bc, y0);
                y1 = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(&iw_fc2[li][(c_idx2 + 8 + k) * N_EMBD + 8])).* , bc, y1);
            }
            inline for (0..4) |k| {
                const hv = h2_3[k];
                const bc: Vec8 = @splat(hv);
                w0v = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(&iw_fc2[li][(c_idx2 + 12 + k) * N_EMBD])).* , bc, w0v);
                w1v = @mulAdd(Vec8, @as(*const [8]f32, @ptrCast(&iw_fc2[li][(c_idx2 + 12 + k) * N_EMBD + 8])).* , bc, w1v);
            }
        }
        const sv: Vec8 = @splat(s_mlp * s_mlp);
        const S0 = (t0 + n0) + (y0 + w0v);
        const S1 = (t1 + n1) + (y1 + w1v);
        const x_lo: Vec8 = @as(*const [8]f32, @ptrCast(x[0..8])).*;
        const x_hi: Vec8 = @as(*const [8]f32, @ptrCast(x[8..16])).*;
        @as(*[8]f32, @ptrCast(x[0..8])).* = @mulAdd(Vec8, S0, sv, x_lo);
        @as(*[8]f32, @ptrCast(x[8..16])).* = @mulAdd(Vec8, S1, sv, x_hi);
    }

    var r0: usize = 0;
    while (r0 + 16 <= lm_pad) : (r0 += 16) {
        mv16_blk16(&x, &iw_lm[r0], lm_pad, @as(*[16]f32, @ptrCast(logits_out[r0..].ptr)));
    }
    if (r0 + 8 <= lm_pad) {
        mv16_blk8(&x, &iw_lm[r0], lm_pad, @as(*[8]f32, @ptrCast(logits_out[r0..].ptr)));
        r0 += 8;
    }
    if (r0 + 4 <= lm_pad) {
        mv16_blk4(&x, &iw_lm[r0], lm_pad, @as(*[4]f32, @ptrCast(logits_out[r0..].ptr)));
        r0 += 4;
    }
}

fn now_s() f64 {
    if (builtin.os.tag == .windows) {
        var freq: winclib.LARGE_INTEGER = undefined;
        var cnt: winclib.LARGE_INTEGER = undefined;
        _ = winclib.QueryPerformanceFrequency(&freq);
        _ = winclib.QueryPerformanceCounter(&cnt);
        return @as(f64, @floatFromInt(cnt.QuadPart)) / @as(f64, @floatFromInt(freq.QuadPart));
    } else {
        var ts: timeclib.timespec = undefined;
        _ = timeclib.clock_gettime(timeclib.CLOCK_MONOTONIC, &ts);
        return @as(f64, @floatFromInt(ts.tv_sec)) + @as(f64, @floatFromInt(ts.tv_nsec)) * 1e-9;
    }
}

inline fn sumsq_acc(acc: *Vec8, s: []const f32) void {
    var i: usize = 0;
    while (i + 8 <= s.len) : (i += 8) {
        const sv: Vec8 = @as(*const [8]f32, @ptrCast(s[i..][0..8])).*;
        acc.* = @mulAdd(Vec8, sv, sv, acc.*);
    }
    while (i < s.len) : (i += 1) {
        acc.* += @as(Vec8, @splat(s[i] * s[i]));
    }
}

// ---------------------------------------------------------------------------
// evaluation
// ---------------------------------------------------------------------------

/// Average nats/char over docs [begin, begin+count) with the currently
/// selected weight tables. Deployment semantics: float activations.
fn eval_docs(begin: usize, count: usize) f64 {
    const saved_act_q = act_q;
    act_q = false;
    defer act_q = saved_act_q;
    var total: f64 = 0;
    var toks: usize = 0;
    var logits: [MAX_CHARS + 1]f32 = undefined;
    for (begin..begin + count) |di| {
        const doc = &docs[di];
        var doc_len: usize = 0;
        while (doc[doc_len] != 0) : (doc_len += 1) {}
        var tokens: [MAX_DOC_LEN + 2]i32 = undefined;
        tokens[0] = @intCast(BOS);
        for (0..doc_len) |i| tokens[i + 1] = char_to_id(doc[i]);
        tokens[doc_len + 1] = @intCast(BOS);
        const n = @min(BLOCK_SIZE, doc_len + 1);
        for (0..n) |pos| {
            gpt_forward(@intCast(tokens[pos]), pos, logits[0..vocab_size], &saved[pos]);
            softmax_fwd_precise(logits[0..vocab_size], vocab_size, saved_probs[pos][0..vocab_size]);
            total += -@log(@as(f64, saved_probs[pos][@intCast(tokens[pos + 1])]) + 1e-30);
            toks += 1;
        }
    }
    return total / @as(f64, @floatFromInt(toks));
}

// ---------------------------------------------------------------------------
// training phases
// ---------------------------------------------------------------------------

fn train_phase(comptime qat: bool, steps: usize, lr: f32, running_loss_init: f32) void {
    weights_q = qat;
    act_q = qat;
    const b1: f32 = 0.9;
    const b2: f32 = 0.999;
    const eps: f32 = 1e-8;
    var running_loss: f32 = running_loss_init;

    const t0 = now_s();
    for (0..steps) |step| {
        if (qat) ternarize_all();

        const doc = &docs[step % n_train];
        var doc_len: usize = 0;
        while (doc[doc_len] != 0) : (doc_len += 1) {}

        var tokens: [MAX_DOC_LEN + 2]i32 = undefined;
        var targets: [BLOCK_SIZE]i32 = undefined;
        tokens[0] = @intCast(BOS);
        for (0..doc_len) |i| tokens[i + 1] = char_to_id(doc[i]);
        tokens[doc_len + 1] = @intCast(BOS);
        const n = @min(BLOCK_SIZE, doc_len + 1);

        var total_loss: f32 = 0;
        var logits: [MAX_CHARS + 1]f32 = undefined;
        for (0..n) |pos| {
            targets[pos] = tokens[pos + 1];
            gpt_forward(@intCast(tokens[pos]), pos, logits[0..vocab_size], &saved[pos]);
            softmax_fwd_precise(logits[0..vocab_size], vocab_size, saved_probs[pos][0..vocab_size]);
            total_loss += -@log(saved_probs[pos][@intCast(targets[pos])] + 1e-30);
        }
        const loss = total_loss / @as(f32, @floatFromInt(n));

        gpt_backward(n, tokens[0..n], targets[0..n]);
        if (qat) mask_all_grads();

        var grad_clip: f32 = 1.0;
        {
            const es = vocab_size * N_EMBD;
            const ps = BLOCK_SIZE * N_EMBD;
            const as = N_EMBD * N_EMBD;
            const ms = MLP_DIM * N_EMBD;
            var gsum: Vec8 = @splat(0);
            sumsq_acc(&gsum, d_wte[0..es]);
            sumsq_acc(&gsum, d_lm_head[0..es]);
            sumsq_acc(&gsum, d_wpe[0..ps]);
            for (0..N_LAYER) |l| {
                sumsq_acc(&gsum, d_attn_wq[l][0..as]);
                sumsq_acc(&gsum, d_attn_wk[l][0..as]);
                sumsq_acc(&gsum, d_attn_wv[l][0..as]);
                sumsq_acc(&gsum, d_attn_wo[l][0..as]);
                sumsq_acc(&gsum, d_mlp_fc1[l][0..ms]);
                sumsq_acc(&gsum, d_mlp_fc2[l][0..ms]);
            }
            const gnorm: f32 = @sqrt(@reduce(.Add, gsum));
            grad_clip = if (gnorm > 1.0) 1.0 / gnorm else 1.0;
        }

        const lr_t: f32 = lr * 0.5 * (1.0 + @cos(@as(f32, @floatCast(math.pi * @as(f64, @floatFromInt(step)) / @as(f64, @floatFromInt(steps))))));
        const stepf: f32 = @as(f32, @floatFromInt(step + 1));
        const b1c: f32 = 1.0 - math.pow(f32, b1, stepf);
        const b2c: f32 = 1.0 - math.pow(f32, b2, stepf);
        const es = vocab_size * N_EMBD;
        const ps = BLOCK_SIZE * N_EMBD;
        const as = N_EMBD * N_EMBD;
        const ms = MLP_DIM * N_EMBD;
        adam_update(wte[0..es], d_wte[0..es], adam_m_wte[0..es], adam_v_wte[0..es], lr_t, b1, b2, eps, b1c, b2c, grad_clip);
        adam_update(wpe[0..ps], d_wpe[0..ps], adam_m_wpe[0..ps], adam_v_wpe[0..ps], lr_t, b1, b2, eps, b1c, b2c, grad_clip);
        adam_update(lm_head[0..es], d_lm_head[0..es], adam_m_lm[0..es], adam_v_lm[0..es], lr_t, b1, b2, eps, b1c, b2c, grad_clip);
        for (0..N_LAYER) |i| {
            adam_update(attn_wq[i][0..as], d_attn_wq[i][0..as], adam_m_wq[i][0..as], adam_v_wq[i][0..as], lr_t, b1, b2, eps, b1c, b2c, grad_clip);
            adam_update(attn_wk[i][0..as], d_attn_wk[i][0..as], adam_m_wk[i][0..as], adam_v_wk[i][0..as], lr_t, b1, b2, eps, b1c, b2c, grad_clip);
            adam_update(attn_wv[i][0..as], d_attn_wv[i][0..as], adam_m_wv[i][0..as], adam_v_wv[i][0..as], lr_t, b1, b2, eps, b1c, b2c, grad_clip);
            adam_update(attn_wo[i][0..as], d_attn_wo[i][0..as], adam_m_wo[i][0..as], adam_v_wo[i][0..as], lr_t, b1, b2, eps, b1c, b2c, grad_clip);
            adam_update(mlp_fc1[i][0..ms], d_mlp_fc1[i][0..ms], adam_m_fc1[i][0..ms], adam_v_fc1[i][0..ms], lr_t, b1, b2, eps, b1c, b2c, grad_clip);
            adam_update(mlp_fc2[i][0..ms], d_mlp_fc2[i][0..ms], adam_m_fc2[i][0..ms], adam_v_fc2[i][0..ms], lr_t, b1, b2, eps, b1c, b2c, grad_clip);
        }

        running_loss = running_loss * 0.99 + loss * 0.01;
        if ((step + 1) % 100 == 0 or step == 0 or step == steps - 1) {
            _ = clib.printf("step %4d / %4d | loss %.4f  (avg %.4f)\n", @as(c_int, @intCast(step + 1)), @as(c_int, @intCast(steps)), @as(f64, loss), @as(f64, running_loss));
        }
    }
    const elapsed = now_s() - t0;
    const name: []const u8 = if (qat) "qat " else "fp32";
    _ = clib.printf("%s train: %.3f s  (%.0f steps/sec)\n", @as([*]const u8, @ptrCast(name.ptr)), elapsed, @as(f64, @floatFromInt(steps)) / elapsed);
}

// ---------------------------------------------------------------------------
// sampling
// ---------------------------------------------------------------------------

fn sample_names(n_samples: usize, temperature: f32) void {
    const mode: []const u8 = if (weights_q) "ternary" else "fp32";
    _ = clib.printf("\ninference (%s weights)\n", @as([*]const u8, @ptrCast(mode.ptr)));
    const inv_t: f32 = 1.0 / temperature;
    for (0..n_samples) |si| {
        var token_id: usize = BOS;
        var buf: [BLOCK_SIZE + 1]u8 = [_]u8{0} ** (BLOCK_SIZE + 1);
        var len: usize = 0;
        for (0..BLOCK_SIZE) |pos| {
            var logits: [LM_PAD_MAX]f32 = undefined;
            gpt_forward_infer(token_id, pos, logits[0..lm_pad]);
            token_id = sample_logits(logits[0..lm_pad], vocab_size, lm_pad, inv_t);
            if (token_id == BOS) break;
            if (token_id < num_uchars) {
                buf[len] = uchars_arr[token_id];
                len += 1;
            }
        }
        buf[len] = 0;
        _ = clib.printf("sample %2d: %s\n", @as(c_int, @intCast(si + 1)), @as([*]const u8, @ptrCast(&buf)));
        for (0..N_LAYER) |li| {
            for (0..BLOCK_SIZE) |tt| {
                @memset(&kv_keys[li][tt], 0);
                @memset(&kv_vals[li][tt], 0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ternary-packed model file
// ---------------------------------------------------------------------------

const FILE_BUF = 1 << 16;

fn put_u32(buf: []u8, cur: *usize, v: u32) void {
    std.mem.writeInt(u32, buf[cur.*..][0..4], v, .little);
    cur.* += 4;
}

fn put_f32(buf: []u8, cur: *usize, v: f32) void {
    std.mem.writeInt(u32, buf[cur.*..][0..4], @bitCast(v), .little);
    cur.* += 4;
}

fn put_f32_arr(buf: []u8, cur: *usize, s: []const f32) void {
    for (s) |v| put_f32(buf, cur, v);
}

fn put_bytes(buf: []u8, cur: *usize, s: []const u8) void {
    @memcpy(buf[cur.*..][0..s.len], s);
    cur.* += s.len;
}

/// One ternary matrix: ternarize, pack, append (scale, nbytes, bytes).
fn put_ternary_matrix(buf: []u8, cur: *usize, w: []const f32, trits: []i8, packed_bytes: []u8) void {
    const info = quant_trits(w, trits);
    const ngroups = ternary.packed_len(w.len);
    ternary.pack5(packed_bytes[0..ngroups], trits);
    put_f32(buf, cur, info.scale);
    put_u32(buf, cur, @intCast(ngroups));
    put_bytes(buf, cur, packed_bytes[0..ngroups]);
}

fn export_packed_model(io: std.Io, path: []const u8) !void {
    var buf: [FILE_BUF]u8 = undefined;
    var cur: usize = 0;
    var pk: [MLP_DIM * N_EMBD / ternary.TRITS_PER_BYTE + 1]u8 = undefined;

    put_bytes(buf[0..], &cur, MAGIC);
    put_u32(buf[0..], &cur, @intCast(N_EMBD));
    put_u32(buf[0..], &cur, @intCast(N_HEAD));
    put_u32(buf[0..], &cur, @intCast(N_LAYER));
    put_u32(buf[0..], &cur, @intCast(BLOCK_SIZE));
    put_u32(buf[0..], &cur, @intCast(MLP_DIM));
    put_u32(buf[0..], &cur, @intCast(vocab_size));
    put_u32(buf[0..], &cur, @intCast(BOS));
    put_u32(buf[0..], &cur, @intCast(num_uchars));
    put_bytes(buf[0..], &cur, uchars_arr[0..num_uchars]);
    put_f32_arr(buf[0..], &cur, wte);
    put_f32_arr(buf[0..], &cur, wpe);
    for (0..N_LAYER) |li| {
        put_ternary_matrix(buf[0..], &cur, attn_wq[li], trits_scratch[0..attn_wq[li].len], pk[0..]);
        put_ternary_matrix(buf[0..], &cur, attn_wk[li], trits_scratch[0..attn_wk[li].len], pk[0..]);
        put_ternary_matrix(buf[0..], &cur, attn_wv[li], trits_scratch[0..attn_wv[li].len], pk[0..]);
        put_ternary_matrix(buf[0..], &cur, attn_wo[li], trits_scratch[0..attn_wo[li].len], pk[0..]);
        put_ternary_matrix(buf[0..], &cur, mlp_fc1[li], trits_scratch[0..mlp_fc1[li].len], pk[0..]);
        put_ternary_matrix(buf[0..], &cur, mlp_fc2[li], trits_scratch[0..mlp_fc2[li].len], pk[0..]);
    }
    put_ternary_matrix(buf[0..], &cur, lm_head, trits_scratch[0..lm_head.len], pk[0..]);

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf[0..cur] });
}

const Reader = struct {
    data: []const u8,
    pos: usize = 0,

    fn take(self: *Reader, n: usize) ?[]const u8 {
        if (self.pos + n > self.data.len) return null;
        const s = self.data[self.pos .. self.pos + n];
        self.pos += n;
        return s;
    }

    fn u32le(self: *Reader) ?u32 {
        const s = self.take(4) orelse return null;
        return std.mem.readInt(u32, s[0..4], .little);
    }

    fn f32le(self: *Reader) ?f32 {
        const v = self.u32le() orelse return null;
        return @bitCast(v);
    }
};

/// Read a packed model file back and rebuild the ternary tables (qw_*).
/// Leaves weights_q untouched; caller decides which tables to use.
fn load_packed_model(io: std.Io, path: []const u8, alloc: std.mem.Allocator) !usize {
    const data = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(FILE_BUF));
    defer alloc.free(data);
    var r = Reader{ .data = data };

    const magic = r.take(MAGIC.len) orelse return error.Truncated;
    if (!std.mem.eql(u8, magic, MAGIC)) return error.BadMagic;
    const h_embd = r.u32le() orelse return error.Truncated;
    const h_head = r.u32le() orelse return error.Truncated;
    const h_layer = r.u32le() orelse return error.Truncated;
    const h_block = r.u32le() orelse return error.Truncated;
    const h_mlp = r.u32le() orelse return error.Truncated;
    const h_vocab = r.u32le() orelse return error.Truncated;
    _ = r.u32le() orelse return error.Truncated; // BOS
    const h_uchars = r.u32le() orelse return error.Truncated;
    if (h_embd != N_EMBD or h_head != N_HEAD or h_layer != N_LAYER or
        h_block != BLOCK_SIZE or h_mlp != MLP_DIM or h_vocab != vocab_size)
    {
        return error.ShapeMismatch;
    }
    if (r.take(h_uchars) == null) return error.Truncated;
    const es = vocab_size * N_EMBD;
    const wte_f = r.take(es * 4) orelse return error.Truncated;
    @memcpy(wte[0..es], std.mem.bytesAsSlice(f32, wte_f));
    const wpe_f = r.take(BLOCK_SIZE * N_EMBD * 4) orelse return error.Truncated;
    @memcpy(wpe[0 .. BLOCK_SIZE * N_EMBD], std.mem.bytesAsSlice(f32, wpe_f));

    var trit_total: usize = 0;
    for (0..N_LAYER) |li| {
        const mats = [_]struct { w: []f32, q: []f32 }{
            .{ .w = attn_wq[li], .q = qw_q[li][0..] },
            .{ .w = attn_wk[li], .q = qw_k[li][0..] },
            .{ .w = attn_wv[li], .q = qw_v[li][0..] },
            .{ .w = attn_wo[li], .q = qw_o[li][0..] },
            .{ .w = mlp_fc1[li], .q = qw_fc1[li][0..] },
            .{ .w = mlp_fc2[li], .q = qw_fc2[li][0..] },
        };
        for (mats) |m| {
            trit_total += try load_one_matrix(&r, m.w.len, m.q);
        }
    }
    trit_total += try load_one_matrix(&r, lm_head.len, qw_lm);
    return trit_total;
}

fn load_one_matrix(r: *Reader, n: usize, qw: []f32) !usize {
    const scale = r.f32le() orelse return error.Truncated;
    const nbytes = r.u32le() orelse return error.Truncated;
    if (nbytes != ternary.packed_len(n)) return error.BadMatrixSize;
    const bytes = r.take(nbytes) orelse return error.Truncated;
    const trits = try gpa.alloc(i8, n);
    defer gpa.free(trits);
    ternary.unpack5(trits, bytes);
    for (qw, trits) |*q, t| {
        q.* = @as(f32, @floatFromInt(t)) * scale;
    }
    return n;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

pub fn main(init: std.process.Init) u8 {
    var data_path: []const u8 = "data/names.txt";
    var pre_steps: usize = 20000;
    var qat_steps: usize = 5000;
    var arg_it = std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa) catch return 1;
    defer arg_it.deinit();
    _ = arg_it.next();
    if (arg_it.next()) |p| data_path = p;
    if (arg_it.next()) |s| pre_steps = std.fmt.parseInt(usize, s, 10) catch pre_steps;
    if (arg_it.next()) |s| qat_steps = std.fmt.parseInt(usize, s, 10) catch qat_steps;

    load_dataset(data_path);

    const doc_order = gpa.alloc(i32, num_docs) catch return 1;
    defer gpa.free(doc_order);
    for (0..num_docs) |i| doc_order[i] = @intCast(i);
    shuffle_ints(doc_order);
    const docs_tmp = gpa.alloc([MAX_DOC_LEN]u8, num_docs) catch return 1;
    defer gpa.free(docs_tmp);
    for (0..num_docs) |i| {
        const idx: usize = @intCast(doc_order[i]);
        @memcpy(&docs_tmp[i], &docs[idx]);
    }
    @memcpy(docs[0..num_docs], docs_tmp);

    // 80/20 train/hold-out split; the val split is never trained on
    n_train = num_docs * 80 / 100;
    const n_val = num_docs - n_train;

    _ = clib.printf("num docs: %d  (train %d, val %d)\n", @as(c_int, @intCast(num_docs)), @as(c_int, @intCast(n_train)), @as(c_int, @intCast(n_val)));
    build_tokenizer();
    _ = clib.printf("vocab size: %d\n", @as(c_int, @intCast(vocab_size)));
    init_params();

    _ = clib.printf("\n== phase 1: fp32 pretrain ==\n");
    train_phase(false, pre_steps, 3e-3, 3.3);

    weights_q = false;
    const fp32_train_loss = eval_docs(0, n_train);
    const fp32_val_loss = eval_docs(n_train, n_val);
    _ = clib.printf("\neval (avg nats/char, float activations)\n");
    _ = clib.printf("  fp32  train %.4f   val %.4f\n", fp32_train_loss, fp32_val_loss);
    infer_pack_weights(false);
    weights_q = false;
    sample_names(10, 0.5);

    // post-training quantization reference: ternarize the fp32 weights and
    // evaluate as-is, without any QAT
    ternarize_all();
    weights_q = true;
    const ptq_val_loss = eval_docs(n_train, n_val);
    _ = clib.printf("  ptq   val %.4f   (ternary, no QAT)\n", ptq_val_loss);

    _ = clib.printf("\n== phase 2: BitNet ternary QAT ==\n");
    reset_adam();
    train_phase(true, qat_steps, 1e-3, @floatCast(ptq_val_loss));

    ternarize_all();
    weights_q = true;
    const qat_train_loss = eval_docs(0, n_train);
    const qat_val_loss = eval_docs(n_train, n_val);
    _ = clib.printf("\neval (avg nats/char, float activations)\n");
    _ = clib.printf("  qat   train %.4f   val %.4f\n", qat_train_loss, qat_val_loss);

    // trit distribution across all quantized matrices
    var hist = [3]usize{ 0, 0, 0 };
    {
        var tr: [MLP_DIM * N_EMBD]i8 = undefined;
        for (0..N_LAYER) |li| {
            const mats = [_][]const f32{ attn_wq[li], attn_wk[li], attn_wv[li], attn_wo[li], mlp_fc1[li], mlp_fc2[li] };
            for (mats) |m| {
                _ = quant_trits(m, tr[0..m.len]);
                for (tr[0..m.len]) |t| hist[@intCast(t + 1)] += 1;
            }
        }
        _ = quant_trits(lm_head, tr[0..lm_head.len]);
        for (tr[0..lm_head.len]) |t| hist[@intCast(t + 1)] += 1;
    }
    const tern_total = hist[0] + hist[1] + hist[2];
    _ = clib.printf("  trits: -1 %.1f%%   0 %.1f%%   +1 %.1f%%\n", 100.0 * @as(f64, @floatFromInt(hist[0])) / @as(f64, @floatFromInt(tern_total)), 100.0 * @as(f64, @floatFromInt(hist[1])) / @as(f64, @floatFromInt(tern_total)), 100.0 * @as(f64, @floatFromInt(hist[2])) / @as(f64, @floatFromInt(tern_total)));

    _ = clib.printf("\n== phase 3: ternary packing + export ==\n");
    infer_pack_weights(true);
    sample_names(10, 0.5);

    const path: []const u8 = "modeldata/bitnet_gpt_packed.bin";
    export_packed_model(init.io, path) catch {
        std.debug.print("failed to write {s}\n", .{path});
        return 1;
    };

    // storage accounting
    const packed_total = (4 * N_LAYER) * ((N_EMBD * N_EMBD + 4) / 5) + (2 * N_LAYER) * ((N_EMBD * MLP_DIM + 4) / 5) + ((vocab_size * N_EMBD + 4) / 5) + (6 * N_LAYER + 1) * 8;
    const n_embed = (vocab_size + BLOCK_SIZE) * N_EMBD;
    const fp32_total = (tern_total + n_embed) * 4;
    const bits2_total = (tern_total + 3) / 4;
    _ = clib.printf("\nstorage\n");
    _ = clib.printf("  ternary weights      : %6d params\n", tern_total);
    _ = clib.printf("  fp32 weights         : %6d bytes\n", fp32_total);
    _ = clib.printf("  naive 2-bit          : %6d bytes\n", bits2_total);
    _ = clib.printf("  packed 5-trits/byte  : %6d bytes  (%.1fx smaller than fp32, %.2f bits/trit incl. scales)\n", packed_total, @as(f64, @floatFromInt(fp32_total)) / @as(f64, @floatFromInt(packed_total)), @as(f64, @floatFromInt(packed_total)) * 8.0 / @as(f64, @floatFromInt(tern_total)));
    _ = clib.printf("  model file %s     : %d bytes (embeddings kept fp32)\n", @as([*]const u8, @ptrCast(path.ptr)), @as(c_int, @intCast(packed_total + n_embed * 4 + num_uchars + 40)));

    // round-trip: reload the packed file, rebuild ternary tables, re-evaluate.
    // the loss must match the pre-export model bit-for-bit.
    const val_before = qat_val_loss;
    const reloaded_trits = load_packed_model(init.io, path, init.gpa) catch {
        std.debug.print("failed to load {s}\n", .{path});
        return 1;
    };
    weights_q = true;
    const val_after = eval_docs(n_train, n_val);
    const delta = @abs(val_before - val_after);
    _ = clib.printf("\npacked round-trip\n");
    _ = clib.printf("  trits reloaded       : %d / %d\n", reloaded_trits, tern_total);
    _ = clib.printf("  val loss pre-export  : %.6f\n", val_before);
    _ = clib.printf("  val loss reloaded    : %.6f\n", val_after);
    if (delta < 1e-9 and reloaded_trits == tern_total) {
        _ = clib.printf("  round-trip: OK (lossless)\n");
        infer_pack_weights(true);
        sample_names(3, 0.5);
    } else {
        _ = clib.printf("  round-trip: MISMATCH (delta %.3e)\n", delta);
        return 1;
    }

    _ = clib.printf("\nsummary\n");
    _ = clib.printf("  fp32 val %.4f | ptq val %.4f | qat val %.4f | qat recovery %.1f%% of fp32->ptq gap\n", fp32_val_loss, ptq_val_loss, qat_val_loss, (1.0 - (qat_val_loss - fp32_val_loss) / (ptq_val_loss - fp32_val_loss + 1e-9)) * 100.0);

    gpa.free(wte);
    gpa.free(d_wte);
    gpa.free(adam_m_wte);
    gpa.free(adam_v_wte);
    gpa.free(wpe);
    gpa.free(d_wpe);
    gpa.free(adam_m_wpe);
    gpa.free(adam_v_wpe);
    gpa.free(lm_head);
    gpa.free(d_lm_head);
    gpa.free(adam_m_lm);
    gpa.free(adam_v_lm);
    gpa.free(qw_lm);
    for (0..N_LAYER) |li| {
        gpa.free(attn_wq[li]);
        gpa.free(d_attn_wq[li]);
        gpa.free(adam_m_wq[li]);
        gpa.free(adam_v_wq[li]);
        gpa.free(attn_wk[li]);
        gpa.free(d_attn_wk[li]);
        gpa.free(adam_m_wk[li]);
        gpa.free(adam_v_wk[li]);
        gpa.free(attn_wv[li]);
        gpa.free(d_attn_wv[li]);
        gpa.free(adam_m_wv[li]);
        gpa.free(adam_v_wv[li]);
        gpa.free(attn_wo[li]);
        gpa.free(d_attn_wo[li]);
        gpa.free(adam_m_wo[li]);
        gpa.free(adam_v_wo[li]);
        gpa.free(mlp_fc1[li]);
        gpa.free(d_mlp_fc1[li]);
        gpa.free(adam_m_fc1[li]);
        gpa.free(adam_v_fc1[li]);
        gpa.free(mlp_fc2[li]);
        gpa.free(d_mlp_fc2[li]);
        gpa.free(adam_m_fc2[li]);
        gpa.free(adam_v_fc2[li]);
    }
    return 0;
}
