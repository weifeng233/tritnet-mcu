//! Packed model container: the D in the framework's export leg.
//!
//! A model file is a little-endian header (per-example magic + dimensions)
//! followed by parameter records, assembled by the `put*` writers and
//! parsed back by `Reader`. The one record type every example shares is the
//! ternary matrix:
//!
//!   f32 absmean scale | u32 packed byte count | packed trits (5 per byte)
//!
//! The payload uses the same 5-trits-per-byte scheme as the weight codec
//! (framework/ternary.zig), so a deployment engine decodes a matrix with one
//! multiply-and-shift loop per trit. fp32 tensors (embeddings, biases,
//! physical parameters) ride along as raw little-endian f32 via `putF32Arr`
//! / `putBytes`.

const std = @import("std");
const ternary = @import("ternary.zig");
const quant = @import("quant.zig");

// ---------------------------------------------------------------------------
// writers
// ---------------------------------------------------------------------------

pub fn putU32(buf: []u8, cur: *usize, v: u32) void {
    std.mem.writeInt(u32, buf[cur.*..][0..4], v, .little);
    cur.* += 4;
}

pub fn putF32(buf: []u8, cur: *usize, v: f32) void {
    std.mem.writeInt(u32, buf[cur.*..][0..4], @bitCast(v), .little);
    cur.* += 4;
}

pub fn putF64(buf: []u8, cur: *usize, v: f64) void {
    std.mem.writeInt(u64, buf[cur.*..][0..8], @bitCast(v), .little);
    cur.* += 8;
}

pub fn putF32Arr(buf: []u8, cur: *usize, s: []const f32) void {
    for (s) |v| putF32(buf, cur, v);
}

pub fn putBytes(buf: []u8, cur: *usize, s: []const u8) void {
    @memcpy(buf[cur.*..][0..s.len], s);
    cur.* += s.len;
}

/// One ternary matrix record: ternarize, pack 5 trits per byte (in place,
/// behind the 12-byte scale/length prefix), append the record.
/// `mask` selects the wired-only absmean variant (NCP wirings); pass null
/// for dense tensors. `trits` must be at least `w.len` long.
pub fn putTernaryMatrix(buf: []u8, cur: *usize, w: []const f32, mask: ?[]const f32, trits: []i8) void {
    const info = if (mask) |mm|
        quant.quantTritsMasked(w, mm, trits)
    else
        quant.quantTrits(w, trits);
    const ngroups = ternary.packed_len(w.len);
    ternary.pack5(buf[cur.* + 8 ..][0..ngroups], trits[0..w.len]);
    putF32(buf, cur, info.scale);
    putU32(buf, cur, @intCast(ngroups));
    cur.* += ngroups;
}

// ---------------------------------------------------------------------------
// reader
// ---------------------------------------------------------------------------

pub const Reader = struct {
    data: []const u8,
    pos: usize = 0,

    pub fn take(self: *Reader, n: usize) ?[]const u8 {
        if (self.pos + n > self.data.len) return null;
        const s = self.data[self.pos .. self.pos + n];
        self.pos += n;
        return s;
    }

    pub fn u32le(self: *Reader) ?u32 {
        const s = self.take(4) orelse return null;
        return std.mem.readInt(u32, s[0..4], .little);
    }

    pub fn f32le(self: *Reader) ?f32 {
        const v = self.u32le() orelse return null;
        return @bitCast(v);
    }

    pub fn f64le(self: *Reader) ?f64 {
        const s = self.take(8) orelse return null;
        return @bitCast(std.mem.readInt(u64, s[0..8], .little));
    }
};

/// Read one ternary matrix record into the deployed table `qw` (scale * trit).
/// Returns the number of trits restored (= `n` on success).
pub fn readMatrix(r: *Reader, n: usize, qw: []f32, alloc: std.mem.Allocator) !usize {
    const scale = r.f32le() orelse return error.Truncated;
    const nbytes = r.u32le() orelse return error.Truncated;
    if (nbytes != ternary.packed_len(n)) return error.BadMatrixSize;
    const bytes = r.take(nbytes) orelse return error.Truncated;
    const trits = try alloc.alloc(i8, n);
    defer alloc.free(trits);
    ternary.unpack5(trits, bytes);
    quant.deployTable(trits, scale, qw);
    return n;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "ternary matrix record round-trips through the container" {
    const n = 37; // not a multiple of the 5-trit block: exercises padding
    var w: [n]f32 = undefined;
    for (&w, 0..) |*x, i| {
        const fi: f32 = @floatFromInt(i);
        x.* = @sin(fi * 0.7) * 2.0;
    }
    var trits: [n]i8 = undefined;
    const info = quant.quantTrits(&w, &trits);

    var buf: [128]u8 = undefined;
    var cur: usize = 0;
    putTernaryMatrix(&buf, &cur, &w, null, &trits);
    try testing.expectEqual(8 + ternary.packed_len(n), cur);

    var r = Reader{ .data = buf[0..cur] };
    var qw: [n]f32 = undefined;
    const got = try readMatrix(&r, n, &qw, testing.allocator);
    try testing.expectEqual(n, got);
    for (qw, trits) |q, t| {
        try testing.expectEqual(@as(f32, @floatFromInt(t)) * info.scale, q);
    }
}

test "u32/f32/f64 writers and reader agree" {
    var buf: [64]u8 = undefined;
    var cur: usize = 0;
    putU32(&buf, &cur, 0xDEADBEEF);
    putF32(&buf, &cur, 1.5);
    putF64(&buf, &cur, -2.25);
    putF32Arr(&buf, &cur, &.{ 0.5, 0.25 });
    putBytes(&buf, &cur, "xy");
    try testing.expectEqual(@as(usize, 4 + 4 + 8 + 8 + 2), cur);

    var r = Reader{ .data = buf[0..cur] };
    try testing.expectEqual(@as(u32, 0xDEADBEEF), r.u32le().?);
    try testing.expectEqual(@as(f32, 1.5), r.f32le().?);
    try testing.expectEqual(@as(f64, -2.25), r.f64le().?);
    try testing.expectEqual(@as(f32, 0.5), r.f32le().?);
    try testing.expectEqual(@as(f32, 0.25), r.f32le().?);
    try testing.expectEqualSlices(u8, "xy", r.take(2).?);
    try testing.expectEqual(@as(?[]const u8, null), r.take(1));
}

test "masked export bakes the sparsity into the trits" {
    const w = [_]f32{ 1.0, 0.0, -1.0, 0.5 };
    const mask = [_]f32{ 1, 0, 1, 0 };
    var trits: [4]i8 = undefined;
    var buf: [32]u8 = undefined;
    var cur: usize = 0;
    putTernaryMatrix(&buf, &cur, &w, &mask, &trits);
    try testing.expectEqual(@as(i8, 0), trits[1]);
    try testing.expectEqual(@as(i8, 0), trits[3]);
}
