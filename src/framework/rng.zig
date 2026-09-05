//! Deterministic RNG shared by the framework's users.
//!
//! One xorshift stream (the 13/7/17 shift triplet) with a uniform f64 in
//! [0, 1) and a Box-Muller Gauss. Each example seeds it once at startup;
//! a given seed produces the identical sequence on every platform, so whole
//! pipelines are reproducible with no external datasets. `state` is public
//! so callers can snapshot/restore it (tests) or isolate a sub-stream
//! (the NCP wiring constructors pin their stream to the wiring seed).

const std = @import("std");
const math = std.math;

/// The generator's 64-bit state. Assign to reseed or to snapshot/restore.
pub var state: u64 = 42;

pub fn next() u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

/// Uniform f64 in [0, 1), 53 random mantissa bits.
pub fn uniform() f64 {
    return @as(f64, @floatFromInt(next() >> 11)) * (1.0 / 9007199254740992.0);
}

/// Box-Muller transform; the first draw is clamped away from log(0).
pub fn gauss(mean: f32, stddev: f32) f32 {
    var uu1 = uniform();
    const uu2 = uniform();
    if (uu1 < 1e-30) uu1 = 1e-30;
    return mean + stddev * @as(f32, @floatCast(math.sqrt(-2.0 * @log(uu1)) * @cos(2.0 * math.pi * uu2)));
}

/// Uniform f32 in [lo, hi).
pub fn range(lo: f32, hi: f32) f32 {
    return lo + @as(f32, @floatCast(uniform())) * (hi - lo);
}

test "sequences are reproducible from a seed" {
    const saved = state;
    defer state = saved;
    state = 0xBEEF;
    var a: [8]u64 = undefined;
    for (&a) |*v| v.* = next();
    state = 0xBEEF;
    for (&a) |*v| try std.testing.expectEqual(v.*, next());
}

test "uniform stays in range and gauss is centered" {
    const saved = state;
    defer state = saved;
    state = 7;
    var sum: f64 = 0;
    for (0..10000) |_| {
        const u = uniform();
        try std.testing.expect(u >= 0.0 and u < 1.0);
        sum += gauss(0, 1);
    }
    try std.testing.expect(@abs(sum / 10000.0) < 0.05);
}

test "range covers [lo, hi)" {
    const saved = state;
    defer state = saved;
    state = 99;
    for (0..1000) |_| {
        const r = range(-2.0, 3.0);
        try std.testing.expect(r >= -2.0 and r < 3.0);
    }
}
