//! Reader for the reference fixture dumps produced by `python/timestral_fixtures.py`.
//!
//! Layout: `u32 magic 'TRFX' | u32 version | u32 count`, then per tensor
//! `u32 name_len | name | u32 ndim | u32 dims[ndim] | f32 data`.

const std = @import("std");

pub const MAGIC = "TRFX";

pub const Tensor = struct {
    dims: []usize,
    data: []f32,
};

pub const Fixture = struct {
    alloc: std.mem.Allocator,
    map: std.StringHashMapUnmanaged(Tensor) = .empty,

    pub fn load(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !Fixture {
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 28)) catch return error.FileNotFound;
        var f = Fixture{ .alloc = alloc };
        errdefer f.deinit();
        defer alloc.free(bytes);
        if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], MAGIC)) return error.BadFixture;
        const count = std.mem.readInt(u32, bytes[8..12], .little);
        var pos: usize = 12;
        for (0..count) |_| {
            const name_len = std.mem.readInt(u32, bytes[pos..][0..4], .little);
            pos += 4;
            const name = bytes[pos..][0..name_len];
            pos += name_len;
            const ndim = std.mem.readInt(u32, bytes[pos..][0..4], .little);
            pos += 4;
            const dims = try alloc.alloc(usize, ndim);
            var n: usize = 1;
            for (0..ndim) |i| {
                dims[i] = std.mem.readInt(u32, bytes[pos..][0..4], .little);
                pos += 4;
                n *= dims[i];
            }
            const data = try alloc.alloc(f32, n);
            for (data) |*v| {
                v.* = @bitCast(std.mem.readInt(u32, bytes[pos..][0..4], .little));
                pos += 4;
            }
            const key = try alloc.dupe(u8, name);
            try f.map.put(alloc, key, .{ .dims = dims, .data = data });
        }
        return f;
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
