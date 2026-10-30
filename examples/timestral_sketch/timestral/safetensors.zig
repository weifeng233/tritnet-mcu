//! Minimal safetensors reader: the fp32 weight source for the Timestral example.
//!
//! A safetensors file is `u64 header_len | JSON header | raw tensor data`, the header
//! mapping each tensor name to `{dtype, shape, data_offsets}`. Only what the deploy leg
//! needs is supported: little-endian F32 tensors, addressed by name and expected shape.
//! Anything else (other dtypes, ragged shapes, truncated files) is an error rather than
//! a silent reinterpretation.

const std = @import("std");

pub const Error = error{
    BadHeaderLength,
    BadHeader,
    TensorNotFound,
    UnsupportedDtype,
    ShapeMismatch,
    Truncated,
    OutOfMemory,
};

pub const Tensor = struct {
    begin: usize,
    end: usize,
    shape: []const usize,
};

pub const File = struct {
    alloc: std.mem.Allocator,
    data: []const u8,
    tensors: std.StringHashMapUnmanaged(Tensor),
    plus_names: std.ArrayListUnmanaged([]const u8),

    pub fn open(alloc: std.mem.Allocator, io: std.Io, path: []const u8) Error!File {
        const raw = std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 30)) catch |e| {
            return switch (e) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.Truncated,
            };
        };
        return openBytes(alloc, raw);
    }

    pub fn openBytes(alloc: std.mem.Allocator, raw: []const u8) Error!File {
        if (raw.len < 8) return error.BadHeaderLength;
        const header_len = std.mem.readInt(u64, raw[0..8], .little);
        if (header_len > raw.len - 8) return error.BadHeaderLength;
        const header = raw[8 .. 8 + @as(usize, @intCast(header_len))];
        const payload = raw[8 + @as(usize, @intCast(header_len)) ..];

        var tensors: std.StringHashMapUnmanaged(Tensor) = .empty;
        var plus_names: std.ArrayListUnmanaged([]const u8) = .empty;

        const parsed = std.json.parseFromSlice(std.json.Value, alloc, header, .{}) catch return error.BadHeader;
        if (parsed.value != .object) return error.BadHeader;
        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            const val = entry.value_ptr.*;
            if (val != .object) continue; // __metadata__
            const o = val.object;
            const dtype = o.get("dtype") orelse continue;
            if (dtype != .string or !std.mem.eql(u8, dtype.string, "F32")) return error.UnsupportedDtype;
            const shape_v = o.get("shape") orelse return error.BadHeader;
            const offs_v = o.get("data_offsets") orelse return error.BadHeader;
            var shape = alloc.alloc(usize, shape_v.array.items.len) catch return error.OutOfMemory;
            for (shape_v.array.items, 0..) |s, i| {
                if (s != .integer or s.integer < 0) return error.BadHeader;
                shape[i] = @intCast(s.integer);
            }
            const begin: usize = @intCast(offs_v.array.items[0].integer);
            const end: usize = @intCast(offs_v.array.items[1].integer);
            if (end > payload.len or begin > end) return error.Truncated;
            const name = entry.key_ptr.*;
            tensors.put(alloc, name, .{ .begin = begin, .end = end, .shape = shape }) catch return error.OutOfMemory;
            plus_names.append(alloc, name) catch return error.OutOfMemory;
        }
        return .{ .alloc = alloc, .data = payload, .tensors = tensors, .plus_names = plus_names };
    }

    /// Copy one F32 tensor into `alloc`, checking its shape.
    pub fn f32s(self: *File, alloc: std.mem.Allocator, name: []const u8, shape: []const usize) Error![]f32 {
        const t = self.tensors.get(name) orelse return error.TensorNotFound;
        if (t.shape.len != shape.len) return error.ShapeMismatch;
        var n: usize = 1;
        for (t.shape, shape) |a, b| {
            if (a != b) return error.ShapeMismatch;
            n *= a;
        }
        if (t.end - t.begin != n * 4) return error.ShapeMismatch;
        const out = alloc.alloc(f32, n) catch return error.OutOfMemory;
        const bytes = self.data[t.begin..t.end];
        for (out, 0..) |*o, i| {
            o.* = @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little));
        }
        return out;
    }

    /// Number of tensors in the file (excluding `__metadata__`).
    pub fn count(self: *const File) usize {
        return self.plus_names.items.len;
    }
};

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "reads a two-tensor file and checks shapes" {
    const header =
        \\{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b":{"dtype":"F32","shape":[1],"data_offsets":[16,20]},"__metadata__":{"format":"pt"}}
    ;
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(testing.allocator);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, @intCast(header.len), .little);
    buf.appendSlice(testing.allocator, &len) catch unreachable;
    buf.appendSlice(testing.allocator, header) catch unreachable;
    const vals = [_]f32{ 1.5, -2.0, 0.25, 3.0 };
    for (vals) |v| {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, @bitCast(v), .little);
        buf.appendSlice(testing.allocator, &b) catch unreachable;
    }
    var f = try File.openBytes(testing.allocator, buf.items);
    defer f.tensors.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), f.count());
    const a = try f.f32s(testing.allocator, "a", &.{ 2, 2 });
    try testing.expectEqualSlices(f32, &vals, a);
    try testing.expectError(error.ShapeMismatch, f.f32s(testing.allocator, "a", &.{4}));
    try testing.expectError(error.TensorNotFound, f.f32s(testing.allocator, "zz", &.{1}));
}
