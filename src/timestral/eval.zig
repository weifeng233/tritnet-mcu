//! Evaluation for the Timestral sketch example: a deterministic stroke corpus and the
//! forecast metrics the report quotes.
//!
//! The corpus mimics what the app receives — a hand-drawn stroke, which is smooth, has a
//! level, a trend, a period or a plateau, and never quite repeats. Everything is generated
//! from the framework's xorshift stream, so a given seed reproduces the same strokes and
//! the same numbers on every machine.

const std = @import("std");
const rng = @import("framework").rng;

pub const N_KINDS = 5;

/// `count` strokes of `t` steps, cycling through the families a sketch canvas produces:
/// a drawn (Catmull-Rom) curve, a sinusoid, a random walk, a trend with smoothed noise,
/// and plateaus. Values sit on the scale of a sketch canvas: roughly [-2, 3].
pub fn strokes(alloc: std.mem.Allocator, count: usize, t: usize, seed: u64) ![]f32 {
    rng.state = seed;
    const out = try alloc.alloc(f32, count * t);
    for (0..count) |s| {
        const kind = s % N_KINDS;
        const row = out[s * t ..][0..t];
        switch (kind) {
            0 => {
                // what a hand draws: a cubic through a few control points
                var knots: [4]f32 = undefined;
                for (&knots) |*k| k.* = rng.range(-1.2, 2.0);
                const a = rng.range(0.7, 1.8);
                const shift = rng.range(-0.4, 0.6);
                for (row, 0..) |*v, i| {
                    const u = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(t - 1)) * 3.0;
                    const seg: usize = @min(@as(usize, @intFromFloat(u)), 2);
                    const f = u - @as(f32, @floatFromInt(seg));
                    const p0 = knots[seg];
                    const p1 = knots[seg + 1];
                    const m0 = if (seg == 0) 0.0 else 0.5 * (knots[seg + 1] - knots[seg - 1]);
                    const m1 = if (seg == 2) 0.0 else 0.5 * (knots[seg + 2] - knots[seg]);
                    const f2 = f * f;
                    const f3 = f2 * f;
                    v.* = shift + a * ((2 * f3 - 3 * f2 + 1) * p0 + (f3 - 2 * f2 + f) * m0 +
                        (-2 * f3 + 3 * f2) * p1 + (f3 - f2) * m1) + rng.gauss(0, 0.015);
                }
            },
            1 => {
                // sinusoid with jittered period and phase
                const period = rng.range(40.0, 110.0);
                const amp = rng.range(0.4, 1.8);
                const phase = rng.range(0.0, 6.28);
                var level: f32 = rng.range(-0.5, 1.5);
                for (row, 0..) |*v, i| {
                    const tt = @as(f32, @floatFromInt(i));
                    level += rng.gauss(0, 0.004);
                    v.* = level + amp * @sin(2.0 * std.math.pi * tt / period + phase) + rng.gauss(0, 0.02);
                }
            },
            2 => {
                // random walk
                var level: f32 = rng.range(-0.5, 0.5);
                for (row) |*v| {
                    level += rng.gauss(0, 0.06);
                    v.* = level;
                }
            },
            3 => {
                // trend + smoothed noise
                const slope = rng.range(-0.004, 0.004);
                const level: f32 = rng.range(0.0, 1.5);
                var smoothed: f32 = 0;
                for (row, 0..) |*v, i| {
                    smoothed = 0.85 * smoothed + 0.15 * rng.gauss(0, 0.09);
                    v.* = level + slope * @as(f32, @floatFromInt(i)) + smoothed;
                }
            },
            else => {
                // plateaus: a step function with noisy edges
                var level: f32 = rng.range(-0.5, 2.0);
                var left: usize = 0;
                for (row) |*v| {
                    if (left == 0) {
                        level = rng.range(-0.5, 2.0);
                        left = 12 + @as(usize, @intFromFloat(rng.range(0.0, 28.0)));
                    }
                    left -= 1;
                    v.* = level + rng.gauss(0, 0.01);
                }
            },
        }
    }
    return out;
}

pub const Metrics = struct {
    /// Mean pinball loss over the 21 quantiles (CRPS-like: the quantile-loss proxy used
    /// by the GIFT-Eval / TIME leaderboards).
    pinball: f64 = 0,
    /// Mean absolute error of the median.
    mae: f64 = 0,
    /// Mean absolute error over the mean absolute first difference of the context.
    mase: f64 = 0,
    /// Empirical coverage of the [0.05, 0.95] band.
    cov90: f64 = 0,
    /// How often the median beat the naive last-value forecast.
    beat_naive: f64 = 0,
};

/// Score `quant` `(rows, h, q)` against `truth` `(rows, h)`, with `ctx` `(rows, t)` and
/// `knots` for the pinball weights.
pub fn score(
    quant: []const f32,
    truth: []const f32,
    ctx: []const f32,
    rows: usize,
    t: usize,
    h: usize,
    knots: []const f32,
    median_idx: usize,
) Metrics {
    var m = Metrics{};
    const q = knots.len;
    var pinball: f64 = 0;
    var mae: f64 = 0;
    var mase: f64 = 0;
    var cov: f64 = 0;
    var beat: usize = 0;
    var n_pin: f64 = 0;
    for (0..rows) |r| {
        var naive_sum: f64 = 0;
        for (1..t) |i| naive_sum += @abs(@as(f64, ctx[r * t + i]) - @as(f64, ctx[r * t + i - 1]));
        const naive = naive_sum / @as(f64, @floatFromInt(t - 1));
        const scale = @max(naive, 1e-8);
        var row_abs: f64 = 0;
        const last = ctx[r * t + t - 1]; // persistence baseline: the last context value
        var naive_abs: f64 = 0;
        for (0..h) |i| {
            const y: f64 = truth[r * h + i];
            for (0..q) |k| {
                const e = y - @as(f64, quant[(r * h + i) * q + k]);
                pinball += @max(@as(f64, knots[k]) * e, (@as(f64, knots[k]) - 1.0) * e);
                n_pin += 1;
            }
            const med = @as(f64, quant[(r * h + i) * q + median_idx]);
            row_abs += @abs(med - y);
            naive_abs += @abs(@as(f64, last) - y);
            const lo = @as(f64, quant[(r * h + i) * q + 1]);
            const hi = @as(f64, quant[(r * h + i) * q + q - 2]);
            if (y >= lo and y <= hi) cov += 1;
        }
        mae += row_abs;
        mase += row_abs / (scale * @as(f64, @floatFromInt(h)));
        if (row_abs < naive_abs) beat += 1;
    }
    m.pinball = pinball / n_pin;
    m.mae = mae / @as(f64, @floatFromInt(rows * h));
    m.mase = mase / @as(f64, @floatFromInt(rows));
    m.cov90 = cov / @as(f64, @floatFromInt(rows * h));
    m.beat_naive = @as(f64, @floatFromInt(beat)) / @as(f64, @floatFromInt(rows));
    return m;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "corpus is deterministic and bounded" {
    const a = try strokes(testing.allocator, 8, 64, 1234);
    defer testing.allocator.free(a);
    const b = try strokes(testing.allocator, 8, 64, 1234);
    defer testing.allocator.free(b);
    try testing.expectEqualSlices(f32, a, b);
    const c = try strokes(testing.allocator, 8, 64, 99);
    defer testing.allocator.free(c);
    try testing.expect(!std.mem.eql(f32, a, c));
    for (a) |v| try testing.expect(std.math.isFinite(v) and @abs(v) < 100);
}

test "pinball of a perfect forecast is zero" {
    const knots = [_]f32{ 0.1, 0.5, 0.9 };
    // three quantiles at the same value: only the median is "perfect"
    var q: [2 * 2 * 3]f32 = undefined;
    const truth = [_]f32{ 1, 2, 3, 4 };
    const ctx = [_]f32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    for (0..2) |r| for (0..2) |i| {
        const v = truth[r * 2 + i];
        q[(r * 2 + i) * 3 + 0] = v - 0.1;
        q[(r * 2 + i) * 3 + 1] = v;
        q[(r * 2 + i) * 3 + 2] = v + 0.1;
    };
    const m = score(&q, &truth, &ctx, 2, 4, 2, &knots, 1);
    try testing.expect(m.pinball > 0 and m.pinball < 0.06);
    try testing.expectApproxEqAbs(@as(f64, 0), m.mae, 1e-6);
}
