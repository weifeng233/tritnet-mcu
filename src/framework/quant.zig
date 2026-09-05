//! BitNet-style ternary quantization: the Q in the framework's QAT.
//!
//! Weight quantizer (BitNet b1.58 style, absmean scales):
//!   scale = mean(|W|) per tensor
//!   trit  = W > scale/2 ? +1 : W < -scale/2 ? -1 : 0
//!   deployed weight = scale * trit
//! Backward uses the straight-through estimator with the gradient masked to
//! |W| < 2*t (t = scale/2), mirroring the BitNetMCU / bitnet_gpt_train.py
//! rule. Activations are quantized per token to 8 bit during QAT training;
//! deployment inference keeps float activations.
//!
//! Sparse (wired) tensors: for the NCP wirings the absmean scale is computed
//! over *wired* entries only, and masked entries are pinned to zero trits --
//! the zeros of the sparse circuit never dilute the scale, and the packed
//! artifact carries the sparsity pattern with no side-car mask.

const std = @import("std");
const math = std.math;

/// absmean quantization result: threshold t = scale/2 and the scale itself.
/// Backprop needs `t` for the STE mask; export needs `scale` for the file.
pub const TernInfo = struct { t: f32, scale: f32 };

/// Quantize a weight tensor to trits {-1, 0, +1} with an absmean scale.
pub fn quantTrits(w: []const f32, trits: []i8) TernInfo {
    var sum: f32 = 0;
    for (w) |x| sum += @abs(x);
    const scale = sum / @as(f32, @floatFromInt(w.len));
    const t = 0.5 * scale;
    for (w, trits) |x, *q| {
        q.* = if (x > t) 1 else if (x < -t) -1 else 0;
    }
    return .{ .t = t, .scale = scale };
}

/// Masked variant: the absmean scale is computed over entries with a nonzero
/// mask only, and masked entries are pinned to zero trits. This is how the
/// NCP wirings survive quantization.
pub fn quantTritsMasked(w: []const f32, mask: []const f32, trits: []i8) TernInfo {
    var sum: f32 = 0;
    var n: usize = 0;
    for (w, 0..) |x, i| {
        if (mask[i] == 0) {
            trits[i] = 0;
            continue;
        }
        sum += @abs(x);
        n += 1;
    }
    const scale = sum / @as(f32, @floatFromInt(n));
    const t = 0.5 * scale;
    for (w, 0..) |x, i| {
        if (mask[i] != 0) trits[i] = if (x > t) 1 else if (x < -t) -1 else 0;
    }
    return .{ .t = t, .scale = scale };
}

/// Fill the deployed weight table (scale * trit) from quantized trits.
pub fn deployTable(trits: []const i8, scale: f32, qw: []f32) void {
    for (qw, trits) |*q, tr| {
        q.* = @as(f32, @floatFromInt(tr)) * scale;
    }
}

/// Quantize + fill the deployed table in one step.
pub fn ternarize(w: []const f32, qw: []f32, trits: []i8) TernInfo {
    const info = quantTrits(w, trits);
    deployTable(trits, info.scale, qw);
    return info;
}

/// Masked quantize + fill (NCP wirings).
pub fn ternarizeMasked(w: []const f32, mask: []const f32, qw: []f32, trits: []i8) TernInfo {
    const info = quantTritsMasked(w, mask, trits);
    deployTable(trits, info.scale, qw);
    return info;
}

/// Straight-through estimator mask: zero gradients where |W| >= 2*t,
/// keeping them only where |W| < 2*t.
pub fn maskGrads(w: []const f32, d: []f32, t: f32) void {
    for (w, d) |x, *g| {
        if (!(@abs(x) < 2 * t)) g.* = 0;
    }
}

/// Per-token 8-bit activation quantization (QAT training only). STE backward
/// is the identity: the forward rounds, the backward passes gradients through.
pub fn actQuant8(v: []f32) void {
    var m: f32 = 1e-5;
    for (v) |x| {
        const a = @abs(x);
        if (a > m) m = a;
    }
    const s: f32 = 127.0 / m;
    for (v) |*x| {
        // |x| <= m so the round lands inside int8 range
        x.* = @round(x.* * s) / s;
    }
}

/// Ternarize `w` (as export/deployment would) and tally the trit histogram.
pub fn histTrits(w: []const f32, trits: []i8, hist: *[3]usize) void {
    _ = quantTrits(w, trits);
    for (trits[0..w.len]) |t| hist[@intCast(t + 1)] += 1;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "absmean quantization matches the definition" {
    // scale = mean(|{-2, 1, 0, 1}|) = 1, t = 0.5
    const w = [_]f32{ -2.0, 1.0, 0.0, 1.0 };
    var trits: [4]i8 = undefined;
    const info = quantTrits(&w, &trits);
    try testing.expectApproxEqAbs(@as(f32, 1.0), info.scale, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), info.t, 1e-6);
    try testing.expectEqualSlices(i8, &.{ -1, 1, 0, 1 }, &trits);
}

test "deploy table is scale * trit" {
    var trits: [4]i8 = .{ -1, 0, 1, 1 };
    var qw: [4]f32 = undefined;
    deployTable(&trits, 0.25, &qw);
    try testing.expectEqualSlices(f32, &.{ -0.25, 0, 0.25, 0.25 }, &qw);
}

test "STE gradient mask keeps |W| < 2t only" {
    // t = 0.5, so the boundary sits at |W| = 1
    const w = [_]f32{ 0.99, 1.0, -1.0, -0.99 };
    var d = [_]f32{ 1, 2, 3, 4 };
    maskGrads(&w, &d, 0.5);
    try testing.expectEqualSlices(f32, &.{ 1, 0, 0, 4 }, &d);
}

test "8-bit activation quantization rescales the max to 127 levels" {
    var v = [_]f32{ 1.0, -0.5, 0.25, 0.0 };
    actQuant8(&v);
    // levels are k/127 with k = round(x * 127); -0.5 * 127 = -63.5 rounds
    // away from zero to -64
    try testing.expectApproxEqAbs(@as(f32, 1.0), v[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -64.0 / 127.0), v[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 32.0 / 127.0), v[2], 1e-6);
    try testing.expectEqual(@as(f32, 0), v[3]);
}

test "masked scale ignores unwired entries and pins their trits" {
    const w = [_]f32{ -2.0, 100.0, 1.0, 1.0 };
    const mask = [_]f32{ 1, 0, 1, 1 };
    var trits: [4]i8 = undefined;
    const info = quantTritsMasked(&w, &mask, &trits);
    // scale over wired entries only: mean(|{-2, 1, 1}|) = 4/3
    try testing.expectApproxEqAbs(4.0 / 3.0, info.scale, 1e-6);
    try testing.expectEqual(@as(i8, 0), trits[1]);
    try testing.expectEqual(@as(i8, -1), trits[0]);
}

test "histTrits tallies every entry exactly once" {
    const w = [_]f32{ -2.0, 1.0, 0.0, 1.0 };
    var trits: [4]i8 = undefined;
    var hist = [3]usize{ 0, 0, 0 };
    histTrits(&w, &trits, &hist);
    try testing.expectEqual(@as(usize, 4), hist[0] + hist[1] + hist[2]);
    try testing.expectEqual(@as(usize, 1), hist[0]);
    try testing.expectEqual(@as(usize, 1), hist[1]);
    try testing.expectEqual(@as(usize, 2), hist[2]);
}
