//! Reader for the reference fixture dumps produced by `examples/timestral_sketch/python/timestral_fixtures.py`.
//!
//! Layout: `u32 magic 'TRFX' | u32 version | u32 count`, then per tensor
//! `u32 name_len | name | u32 ndim | u32 dims[ndim] | f32 data`.
//!
//! Every length in the file is attacker-controlled as far as this reader is concerned
//! -- a truncated download or a half-written fixture is the realistic failure, not a
//! hostile one -- so each field is bounds-checked against the bytes that remain before
//! it is used. A bad fixture is an error, never a panic.

const std = @import("std");

pub const MAGIC = "TRFX";
pub const VERSION: u32 = 1;
/// Sanity bound: the real dumps have at most 4 dimensions; anything beyond this is a
/// corrupt length, not a tensor.
pub const MAX_DIMS: usize = 8;

pub const Tensor = struct {
    dims: []usize,
    data: []f32,
};

pub const Fixture = struct {
    alloc: std.mem.Allocator,
    map: std.StringHashMapUnmanaged(Tensor) = .empty,

    pub fn load(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !Fixture {
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 28)) catch return error.FileNotFound;
        defer alloc.free(bytes);
        var f = Fixture{ .alloc = alloc };
        errdefer f.deinit();
        try parse(&f, bytes);
        return f;
    }

    /// Parse in-memory fixture bytes. Split out from `load` so tests can feed it a
    /// malformed buffer without touching the filesystem.
    pub fn parse(f: *Fixture, bytes: []const u8) !void {
        const alloc = f.alloc;
        var r = Cursor{ .data = bytes };
        const magic = r.take(4) orelse return error.BadFixture;
        if (!std.mem.eql(u8, magic, MAGIC)) return error.BadFixture;
        const version = r.readU32() orelse return error.BadFixture;
        if (version != VERSION) return error.BadFixtureVersion;
        const count = r.readU32() orelse return error.BadFixture;

        for (0..count) |_| {
            const name_len = r.readU32() orelse return error.BadFixture;
            const name = r.take(name_len) orelse return error.BadFixture;
            const ndim = r.readU32() orelse return error.BadFixture;
            if (ndim > MAX_DIMS) return error.BadFixture;
            const dims = try alloc.alloc(usize, ndim);
            errdefer alloc.free(dims);
            var n: usize = 1;
            for (0..ndim) |i| {
                dims[i] = r.readU32() orelse return error.BadFixture;
                n = std.math.mul(usize, n, dims[i]) catch return error.BadFixture;
            }
            // the declared element count has to fit in the bytes that remain
            const need = std.math.mul(usize, n, 4) catch return error.BadFixture;
            if (need > r.remaining()) return error.BadFixture;
            const data = try alloc.alloc(f32, n);
            errdefer alloc.free(data);
            for (data) |*v| v.* = @bitCast(r.readU32().?);
            const key = try alloc.dupe(u8, name);
            errdefer alloc.free(key);
            try f.map.put(alloc, key, .{ .dims = dims, .data = data });
        }
        if (r.remaining() != 0) return error.TrailingBytes;
    }

    pub fn deinit(self: *Fixture) void {
        var it = self.map.iterator();
        while (it.next()) |e| {
            self.alloc.free(e.key_ptr.*);
            self.alloc.free(e.value_ptr.dims);
            self.alloc.free(e.value_ptr.data);
        }
        self.map.deinit(self.alloc);
    }

    pub fn get(self: *const Fixture, name: []const u8) !Tensor {
        return self.map.get(name) orelse error.TensorNotFound;
    }
};

/// Bounds-checked little-endian cursor. Every accessor returns null instead of
/// panicking when the buffer is exhausted.
const Cursor = struct {
    data: []const u8,
    pos: usize = 0,

    fn remaining(self: *const Cursor) usize {
        return self.data.len - self.pos;
    }

    fn take(self: *Cursor, n: usize) ?[]const u8 {
        if (n > self.remaining()) return null;
        const s = self.data[self.pos..][0..n];
        self.pos += n;
        return s;
    }

    fn readU32(self: *Cursor) ?u32 {
        const s = self.take(4) orelse return null;
        return std.mem.readInt(u32, s[0..4], .little);
    }
};

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

fn buildFixture(alloc: std.mem.Allocator, version: u32, dims: []const u32, values: []const f32) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(alloc);
    try out.appendSlice(alloc, MAGIC);
    try appendU32(&out, alloc, version);
    try appendU32(&out, alloc, 1); // one tensor
    try appendU32(&out, alloc, 4); // name length
    try out.appendSlice(alloc, "test");
    try appendU32(&out, alloc, @intCast(dims.len));
    for (dims) |d| try appendU32(&out, alloc, d);
    for (values) |v| try appendU32(&out, alloc, @bitCast(v));
    return out.toOwnedSlice(alloc);
}

fn appendU32(out: *std.ArrayListUnmanaged(u8), alloc: std.mem.Allocator, v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try out.appendSlice(alloc, &b);
}

test "round-trips a well-formed fixture" {
    const bytes = try buildFixture(testing.allocator, VERSION, &.{ 2, 3 }, &.{ 1.5, -2.0, 0.25, 3.0, 4.0, 5.0 });
    defer testing.allocator.free(bytes);
    var f = Fixture{ .alloc = testing.allocator };
    defer f.deinit();
    try f.parse(bytes);
    const t = try f.get("test");
    try testing.expectEqualSlices(usize, &.{ 2, 3 }, t.dims);
    try testing.expectEqualSlices(f32, &.{ 1.5, -2.0, 0.25, 3.0, 4.0, 5.0 }, t.data);
}

test "rejects a fixture from a different version" {
    const bytes = try buildFixture(testing.allocator, 2, &.{1}, &.{0.0});
    defer testing.allocator.free(bytes);
    var f = Fixture{ .alloc = testing.allocator };
    defer f.deinit();
    try testing.expectError(error.BadFixtureVersion, f.parse(bytes));
}

test "rejects a truncated fixture instead of reading past the end" {
    const bytes = try buildFixture(testing.allocator, VERSION, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer testing.allocator.free(bytes);
    // Claim twice the bytes that are actually there: the element-count check has to
    // catch this before any read goes out of bounds.
    var f = Fixture{ .alloc = testing.allocator };
    defer f.deinit();
    try testing.expectError(error.BadFixture, f.parse(bytes[0 .. bytes.len - 8]));

    // and a length field that is larger than the remaining buffer
    var g = Fixture{ .alloc = testing.allocator };
    defer g.deinit();
    var bad: [64]u8 = undefined;
    @memcpy(bad[0..8], "TRFX\x01\x00\x00\x00");
    std.mem.writeInt(u32, bad[8..12], 1, .little);
    std.mem.writeInt(u32, bad[12..16], 0xFFFF, .little); // name_len far too large
    try testing.expectError(error.BadFixture, g.parse(bad[0..32]));
}
