//! Adam optimizer + cosine learning-rate schedule shared by the fp32
//! baseline and every example. `update` is the 8-wide SIMD kernel (AVX2 /
//! NEON via @Vector) with a scalar tail, so any parameter length works.

const std = @import("std");
const math = std.math;

pub const BETA1: f32 = 0.9;
pub const BETA2: f32 = 0.999;
pub const EPS: f32 = 1e-8;

const Vec8 = @Vector(8, f32);

/// One Adam step over the parameter vector `p`. The gradient `g` is scaled
/// by `gscale` (global-norm clipping is folded in here), consumed to update
/// the moments `m`/`v`, and zeroed so the next backward pass starts clean.
pub fn update(p: []f32, g: []f32, m: []f32, v: []f32, lr: f32, b1: f32, b2: f32, eps: f32, b1c: f32, b2c: f32, gscale: f32) void {
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

/// Bias-correction complements (1 - beta^t) at optimizer step `step`
/// (0-based; the first step is t = 1).
pub fn biasCorr(step: usize) struct { b1c: f32, b2c: f32 } {
    const t: f32 = @floatFromInt(step + 1);
    return .{
        .b1c = 1.0 - math.pow(f32, BETA1, t),
        .b2c = 1.0 - math.pow(f32, BETA2, t),
    };
}

/// Cosine decay from `lr` at step 0 down to ~0 at step `steps`.
pub fn cosineLr(lr: f32, step: usize, steps: usize) f32 {
    return @floatCast(lr * 0.5 * (1.0 + @cos(math.pi * @as(f64, @floatFromInt(step)) / @as(f64, @floatFromInt(steps)))));
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "adam descends a quadratic and zeroes the gradient" {
    var p = [_]f32{ 2.0, -2.0, 0.5, 0.25 } ++ [_]f32{0.0} ** 12;
    var g = [_]f32{0.0} ** 16;
    var m = [_]f32{0.0} ** 16;
    var v = [_]f32{0.0} ** 16;
    const bc = biasCorr(0);
    // d/dp (p - 1)^2 = 2(p - 1): every step pulls p toward 1
    for (&p, &g) |*pp, *gg| gg.* = 2.0 * (pp.* - 1.0);
    update(&p, &g, &m, &v, 0.1, BETA1, BETA2, EPS, bc.b1c, bc.b2c, 1.0);
    try testing.expect(p[0] < 2.0 and p[1] > -2.0);
    try testing.expect(p[2] > 0.5 and p[3] > 0.25);
    for (g) |gg| try testing.expectEqual(@as(f32, 0), gg);
}

test "cosine schedule spans lr to zero" {
    try testing.expectApproxEqAbs(@as(f32, 1.0), cosineLr(1.0, 0, 100), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), cosineLr(1.0, 50, 100), 1e-5);
    try testing.expect(cosineLr(1.0, 100, 100) < 1e-6);
}

test "bias correction matches 1 - beta^t" {
    const bc = biasCorr(0);
    try testing.expectApproxEqAbs(@as(f32, 1.0 - BETA1), bc.b1c, 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 1.0 - BETA2), bc.b2c, 1e-7);
}
