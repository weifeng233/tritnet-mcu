//! Ternary packing: 5 trits {-1, 0, 1} per byte.
//!
//! Scheme from https://compilade.net/blog/ternary-packing (the same scheme
//! llama.cpp uses for BitNet b1.58 / TriLM weights). Since 3^5 = 243 <= 256,
//! five trits fit in one byte, giving 1.6 bits per weight against the
//! log2(3) ~= 1.585 information-theoretic floor.
//!
//! Encode: shift trits to {0, 1, 2}, fold them into a base-3 integer with
//! Horner's method, then rescale 0..242 onto the full byte range with
//! CEILING division by 243. The ceiling matters: it cancels the truncation
//! error when digits are later extracted from the top.
//!
//! Decode: division-free and SIMD-friendly. Multiplying the packed byte by 3
//! makes it a 10-bit value whose top bits (>> 8) are the next most-significant
//! trit; masking with 0xFF keeps the remainder. Five rounds empty the byte.

const std = @import("std");

pub const TRITS_PER_BYTE = 5;
/// 3^5, the number of distinct 5-trit blocks
pub const BLOCKS = 243;

/// Number of bytes needed to pack `n` trits (zero-padded to a full block).
pub fn packed_len(n: usize) usize {
    return (n + TRITS_PER_BYTE - 1) / TRITS_PER_BYTE;
}

/// Pack groups of 5 trits into bytes. `dst.len` groups are consumed from
/// `src`; trits past `src.len` are packed as 0. Every source trit must be
/// -1, 0 or 1 (safety-checked in debug builds).
pub fn pack5(dst: []u8, src: []const i8) void {
    var g: usize = 0;
    while (g < dst.len) : (g += 1) {
        var b: u32 = 0;
        inline for (0..TRITS_PER_BYTE) |i| {
            const idx = g * TRITS_PER_BYTE + i;
            const t: i32 = if (idx < src.len) src[idx] else 0;
            std.debug.assert(t >= -1 and t <= 1);
            b = b * 3 + @as(u32, @intCast(t + 1));
        }
        dst[g] = @intCast((b * 256 + BLOCKS - 1) / BLOCKS);
    }
}

/// Unpack bytes into trits. Peels digits off the top of each byte with a
/// multiply-by-3 / shift pair -- no division or modulo. Values past the
/// last destination trit are discarded, so padding round-trips safely.
pub fn unpack5(dst: []i8, src: []const u8) void {
    var i: usize = 0;
    var g: usize = 0;
    while (i < dst.len) : (g += 1) {
        var b: u16 = src[g];
        inline for (0..TRITS_PER_BYTE) |_| {
            if (i < dst.len) {
                b *= 3;
                const t: u16 = b >> 8;
                b &= 0xFF;
                dst[i] = @as(i8, @intCast(t)) - 1;
                i += 1;
            }
        }
    }
}

test "round-trips all 243 possible 5-trit blocks" {
    // Every base-3 value 0..242 maps to one digit pattern; exhaustively
    // check encode->decode restores it (mirrors the blog's verification).
    var trits: [TRITS_PER_BYTE]i8 = undefined;
    var back: [TRITS_PER_BYTE]i8 = undefined;
    var packed_buf: [1]u8 = undefined;
    var checked: usize = 0;
    for (0..BLOCKS) |v| {
        var r = v;
        var i: usize = TRITS_PER_BYTE;
        while (i > 0) {
            i -= 1;
            trits[i] = @intCast(@as(i32, @intCast(r % 3)) - 1);
            r /= 3;
        }
        pack5(packed_buf[0..], trits[0..]);
        unpack5(back[0..], packed_buf[0..]);
        try std.testing.expectEqualSlices(i8, trits[0..], back[0..]);
        // packing must stay within one byte
        try std.testing.expect(packed_buf[0] <= 255);
        checked += 1;
    }
    try std.testing.expectEqual(BLOCKS, checked);
}

test "round-trips streams of awkward lengths" {
    const lengths = [_]usize{ 0, 1, 2, 4, 5, 6, 7, 63, 64, 1024, 3072 };
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();
    for (lengths) |n| {
        const src = try std.testing.allocator.alloc(i8, n);
        defer std.testing.allocator.free(src);
        const back = try std.testing.allocator.alloc(i8, n);
        defer std.testing.allocator.free(back);
        for (src) |*t| t.* = rand.intRangeAtMost(i2, -1, 1);
        const pk = try std.testing.allocator.alloc(u8, packed_len(n));
        defer std.testing.allocator.free(pk);
        pack5(pk, src);
        unpack5(back, pk);
        try std.testing.expectEqualSlices(i8, src, back);
    }
}

test "packing density is 5 trits per byte" {
    try std.testing.expectEqual(@as(usize, 0), packed_len(0));
    try std.testing.expectEqual(@as(usize, 1), packed_len(5));
    try std.testing.expectEqual(@as(usize, 2), packed_len(6));
    try std.testing.expectEqual(@as(usize, 615), packed_len(3072));
}

test "distinct blocks pack to distinct bytes" {
    // the ceiling rescale must stay injective over 0..242
    var seen = [_]bool{false} ** 256;
    var trits: [TRITS_PER_BYTE]i8 = undefined;
    var pk: [1]u8 = undefined;
    for (0..BLOCKS) |v| {
        var r = v;
        var i: usize = TRITS_PER_BYTE;
        while (i > 0) {
            i -= 1;
            trits[i] = @intCast(@as(i32, @intCast(r % 3)) - 1);
            r /= 3;
        }
        pack5(pk[0..], trits[0..]);
        try std.testing.expect(!seen[pk[0]]);
        seen[pk[0]] = true;
    }
}

test "golden vectors: the worked examples in docs/ARTIFACT-FORMAT.md" {
    // These are published in the format document so an independent implementation
    // can be checked against this one without reading Zig. If the codec ever
    // changes, the document and this test change together.
    const Case = struct { trits: [5]i8, byte: u8 };
    const cases = [_]Case{
        .{ .trits = .{ 1, 0, -1, 0, 1 }, .byte = 0xCD }, // base-3 194 -> ceil(194*256/243) = 205
        .{ .trits = .{ -1, -1, -1, -1, -1 }, .byte = 0x00 }, // 0, the low end
        .{ .trits = .{ 1, 1, 1, 1, 1 }, .byte = 0xFF }, // 242, the high end
        .{ .trits = .{ 1, 1, 1, 1, -1 }, .byte = 0xFD }, // 240 -> ceil(240*256/243) = 253
    };
    var pk: [1]u8 = undefined;
    for (cases) |c| {
        pack5(pk[0..], c.trits[0..]);
        try std.testing.expectEqual(c.byte, pk[0]);
        var back: [5]i8 = undefined;
        unpack5(back[0..], pk[0..]);
        try std.testing.expectEqualSlices(i8, c.trits[0..], back[0..]);
    }
}
