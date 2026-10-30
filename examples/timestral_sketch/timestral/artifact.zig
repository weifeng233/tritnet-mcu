//! Packed Timestral artifact: the deployable form of the sketch-a-series model.
//!
//! One file carries everything the engine needs — the architecture config, the quantile
//! knots, every fp32 tensor (norms, biases, the forecast query), and every projection as
//! a ternary matrix with one absmean scale per output row (5 trits per byte inside the
//! record). No safetensors reader, no PyTorch: `timestral-cli serve` loads this and
//! nothing else.
//!
//!   "TRTL" u32 version
//!   u32 patch_size d_model num_heads mlp_hidden n_layers n_knots block_size rope_dim
//!   f32 width_factor norm_eps rope_theta min_scale
//!   u8 use_asinh attend_missing
//!   u8 schedule[n_layers] f32 knots[n_knots]
//!   per tensor, in the canonical order below:
//!     u8 0 | u32 n | f32 values[n]                                (fp32)
//!     u8 1 | u32 rows | u32 cols | f32 scales[rows] | u32 nbytes | u8 trits[nbytes]

const std = @import("std");
const model = @import("model.zig");
const ternary = @import("framework").ternary;

pub const MAGIC = "TRTL";
pub const VERSION: u32 = 2;

pub const Reader = struct {
    data: []const u8,
    pos: usize = 0,

    pub fn take(self: *Reader, n: usize) ![]const u8 {
        if (self.pos + n > self.data.len) return error.Truncated;
        const s = self.data[self.pos .. self.pos + n];
        self.pos += n;
        return s;
    }

    pub fn u8v(self: *Reader) !u8 {
        const s = try self.take(1);
        return s[0];
    }

    pub fn u32v(self: *Reader) !u32 {
        return std.mem.readInt(u32, (try self.take(4))[0..4], .little);
    }

    pub fn f32v(self: *Reader) !f32 {
        return @bitCast(try self.u32v());
    }
};

pub fn pushU8(list: *std.ArrayListUnmanaged(u8), alloc: std.mem.Allocator, v: u8) !void {
    try list.append(alloc, v);
}

pub fn pushU32(list: *std.ArrayListUnmanaged(u8), alloc: std.mem.Allocator, v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try list.appendSlice(alloc, &b);
}

pub fn pushF32(list: *std.ArrayListUnmanaged(u8), alloc: std.mem.Allocator, v: f32) !void {
    try pushU32(list, alloc, @bitCast(v));
}

const Encoder = struct {
    list: *std.ArrayListUnmanaged(u8),
    alloc: std.mem.Allocator,

    fn vec(self: *Encoder, v: *const []f32) !void {
        try pushU32(self.list, self.alloc, @intCast(v.len));
        for (v.*) |x| try pushF32(self.list, self.alloc, x);
    }

    fn matrix(self: *Encoder, w: *model.Matrix) !void {
        if (!w.tern) {
            // fp32 matrix: kept dense (a projection that must not be ternarized)
            try pushU8(self.list, self.alloc, 0);
            try pushU32(self.list, self.alloc, @intCast(w.rows));
            try pushU32(self.list, self.alloc, @intCast(w.cols));
            for (w.f32s) |x| try pushF32(self.list, self.alloc, x);
            return;
        }
        try pushU8(self.list, self.alloc, 1);
        try pushU32(self.list, self.alloc, @intCast(w.rows));
        try pushU32(self.list, self.alloc, @intCast(w.cols));
        for (w.scales) |s| try pushF32(self.list, self.alloc, s);
        const nbytes = ternary.packed_len(w.rows * w.cols);
        try pushU32(self.list, self.alloc, @intCast(nbytes));
        const start = self.list.items.len;
        try self.list.appendNTimes(self.alloc, 0, nbytes);
        ternary.pack5(self.list.items[start..][0..nbytes], w.trits);
    }
};

const Decoder = struct {
    r: *Reader,
    alloc: std.mem.Allocator,

    fn vec(self: *Decoder, out: *[]f32) !void {
        const n = try self.r.u32v();
        const dst = try self.alloc.alloc(f32, n);
        for (dst) |*x| x.* = try self.r.f32v();
        out.* = dst;
    }

    fn matrix(self: *Decoder, w: *model.Matrix) !void {
        const kind = try self.r.u8v();
        if (kind == 0) {
            const rows = try self.r.u32v();
            const cols = try self.r.u32v();
            const dst = try self.alloc.alloc(f32, @as(usize, rows) * cols);
            for (dst) |*x| x.* = try self.r.f32v();
            w.* = .{ .rows = rows, .cols = cols, .f32s = dst };
            return;
        }
        if (kind != 1) {
            std.debug.print("artifact: bad record kind {d} at offset {d}\n", .{ kind, self.r.pos - 1 });
            return error.BadRecord;
        }
        const rows = try self.r.u32v();
        const cols = try self.r.u32v();
        const scales = try self.alloc.alloc(f32, rows);
        for (scales) |*s| s.* = try self.r.f32v();
        const nbytes = try self.r.u32v();
        if (nbytes != ternary.packed_len(@as(usize, rows) * cols)) return error.BadRecord;
        const payload = try self.r.take(nbytes);
        const trits = try self.alloc.alloc(i8, @as(usize, rows) * cols);
        ternary.unpack5(trits, payload);
        w.* = .{
            .rows = rows,
            .cols = cols,
            .tern = true,
            .trits = trits,
            .scales = scales,
        };
    }
};

/// Canonical visit order: patch block, forecast query, layers, out norm, head.
fn walk(m: *model.Model, v: anytype) !void {
    try v.matrix(&m.patch.hidden);
    try v.vec(&m.patch.hidden_b);
    try v.matrix(&m.patch.out);
    try v.vec(&m.patch.out_b);
    try v.matrix(&m.patch.res);
    try v.vec(&m.patch.res_b);
    try v.vec(&m.forecast_query);
    for (m.layers) |*l| {
        // only the attention axes this layer kind runs; the other slot is never loaded
        for (l.attn[0..l.kind.axes().len], 0..) |*a, j| {
            try v.matrix(&a.wqkv);
            try v.matrix(&a.wO);
            try v.matrix(&a.gate);
            try v.vec(&l.attn_norm[j]);
        }
        try v.matrix(&l.w12);
        try v.matrix(&l.w3);
        try v.vec(&l.mlp_norm);
    }
    try v.vec(&m.out_norm);
    try v.matrix(&m.head.hidden);
    try v.matrix(&m.head.out);
    try v.matrix(&m.head.res);
}

/// Serialize `m` and write it to `path`. Returns the number of bytes written.
pub fn save(m: *model.Model, alloc: std.mem.Allocator, io: std.Io, path: []const u8) !usize {
    const cfg = m.cfg;
    var list: std.ArrayListUnmanaged(u8) = .empty;
    defer list.deinit(alloc);
    try list.appendSlice(alloc, MAGIC);
    try pushU32(&list, alloc, VERSION);
    try pushU32(&list, alloc, @intCast(cfg.patch_size));
    try pushU32(&list, alloc, @intCast(cfg.d_model));
    try pushU32(&list, alloc, @intCast(cfg.num_heads));
    try pushU32(&list, alloc, @intCast(cfg.mlp_hidden));
    try pushU32(&list, alloc, @intCast(cfg.n_layers));
    try pushU32(&list, alloc, @intCast(cfg.n_knots));
    try pushU32(&list, alloc, @intCast(cfg.block_size));
    try pushU32(&list, alloc, @intCast(cfg.rope_dim));
    try pushF32(&list, alloc, cfg.width_factor);
    try pushF32(&list, alloc, cfg.norm_eps);
    try pushF32(&list, alloc, cfg.rope_theta);
    try pushF32(&list, alloc, cfg.min_scale);
    try pushF32(&list, alloc, cfg.width_scale);
    try pushU8(&list, alloc, @intFromBool(cfg.use_asinh));
    try pushU8(&list, alloc, @intFromBool(cfg.attend_missing));
    for (cfg.schedule[0..cfg.n_layers]) |k| try pushU8(&list, alloc, @intFromEnum(k));
    for (cfg.knots[0..cfg.n_knots]) |k| try pushF32(&list, alloc, k);

    var enc = Encoder{ .list = &list, .alloc = alloc };
    try walk(m, &enc);

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = list.items });
    return list.items.len;
}

/// Load a packed artifact. The returned model owns its own arena.
pub fn load(backing: std.mem.Allocator, io: std.Io, path: []const u8) !model.Model {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, backing, .limited(1 << 30)) catch return error.Truncated;
    defer backing.free(bytes);
    var r = Reader{ .data = bytes };
    const magic = try r.take(4);
    if (!std.mem.eql(u8, magic, MAGIC)) return error.BadMagic;
    if (try r.u32v() != VERSION) return error.BadVersion;
    var cfg = model.Config{};
    cfg.patch_size = try r.u32v();
    cfg.d_model = try r.u32v();
    cfg.num_heads = try r.u32v();
    cfg.mlp_hidden = try r.u32v();
    cfg.n_layers = try r.u32v();
    cfg.n_knots = try r.u32v();
    cfg.block_size = try r.u32v();
    cfg.rope_dim = try r.u32v();
    cfg.width_factor = try r.f32v();
    cfg.norm_eps = try r.f32v();
    cfg.rope_theta = try r.f32v();
    cfg.min_scale = try r.f32v();
    cfg.width_scale = try r.f32v();
    cfg.use_asinh = (try r.u8v()) != 0;
    cfg.attend_missing = (try r.u8v()) != 0;
    if (cfg.n_layers == 0 or cfg.n_layers > model.MAX_LAYERS) return error.BadConfig;
    if (cfg.n_knots > model.MAX_KNOTS) return error.BadConfig;
    cfg.schedule = [_]model.Kind{.time} ** model.MAX_LAYERS;
    for (0..cfg.n_layers) |i| cfg.schedule[i] = @enumFromInt(try r.u8v());
    for (0..cfg.n_knots) |i| cfg.knots[i] = try r.f32v();

    var m = try model.Model.create(backing, cfg);
    errdefer m.deinit();
    var dec = Decoder{ .r = &r, .alloc = m.arenaAlloc() };
    try walk(&m, &dec);
    // The canonical order is the schema, so a file that ends early is corrupt and a
    // file with bytes left over was written by something that does not agree with
    // this reader. Silently ignoring the tail hides exactly that disagreement.
    if (r.pos != r.data.len) return error.TrailingBytes;
    return m;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Fill one matrix with a deterministic pattern. Ternary matrices exercise the
/// kind-1 record; fp32 matrices exercise the kind-0 one.
fn fillMatrix(alloc: std.mem.Allocator, w: *model.Matrix, rows: usize, cols: usize, tern: bool, seed: f32) !void {
    w.rows = rows;
    w.cols = cols;
    w.tern = tern;
    if (!tern) {
        w.f32s = try alloc.alloc(f32, rows * cols);
        for (w.f32s, 0..) |*v, i| v.* = seed + @as(f32, @floatFromInt(i % 17)) * 0.125;
        return;
    }
    w.trits = try alloc.alloc(i8, rows * cols);
    w.scales = try alloc.alloc(f32, rows);
    for (w.trits, 0..) |*t, i| t.* = @intCast(@as(i32, @intCast((i * 7) % 3)) - 1);
    for (w.scales, 0..) |*s, i| s.* = seed + @as(f32, @floatFromInt(i)) * 0.5;
}

fn fillVec(alloc: std.mem.Allocator, out: *[]f32, n: usize, seed: f32) !void {
    out.* = try alloc.alloc(f32, n);
    for (out.*, 0..) |*v, i| v.* = seed - @as(f32, @floatFromInt(i)) * 0.25;
}

test "record layout round-trips a small model without a checkpoint" {
    // The published format is claims in a document; this test is the document
    // executed. It builds a model small enough to fit in a test, writes every
    // record kind (fp32 vector, dense matrix, ternary matrix) in the canonical
    // order, reads it back and compares values. If a writer or reader changes the
    // record layout, this fails here -- before anyone writes a reader from the doc.
    var cfg = model.Config{
        .patch_size = 4,
        .d_model = 8,
        .num_heads = 2,
        .mlp_hidden = 16,
        .n_layers = 2,
        .n_knots = 5,
        .block_size = 4,
        .rope_dim = 4,
        .width_scale = 1.25,
        .use_asinh = false,
        .attend_missing = true,
    };
    // one time-axis layer and one mixed layer, so the axis-dependent record count
    // is exercised too
    cfg.schedule[0] = .time;
    cfg.schedule[1] = .time_variate;
    for (0..cfg.n_knots) |i| cfg.knots[i] = 0.01 + 0.2 * @as(f32, @floatFromInt(i));

    var m = try model.Model.create(testing.allocator, cfg);
    defer m.deinit();
    const alloc = m.arenaAlloc();

    const d = cfg.d_model;
    try fillMatrix(alloc, &m.patch.hidden, 4, 2 * cfg.patch_size, true, 0.5);
    try fillVec(alloc, &m.patch.hidden_b, 4, 0.25);
    try fillMatrix(alloc, &m.patch.out, d, 4, true, 0.75);
    try fillVec(alloc, &m.patch.out_b, d, 0.125);
    // one dense (never ternarized) matrix, so both record kinds are covered
    try fillMatrix(alloc, &m.patch.res, d, 2 * cfg.patch_size, false, 1.0);
    try fillVec(alloc, &m.patch.res_b, d, 2.0);
    try fillVec(alloc, &m.forecast_query, d, 3.0);

    for (m.layers, 0..) |*l, li| {
        for (l.attn[0..l.kind.axes().len], 0..) |*a, ai| {
            try fillMatrix(alloc, &a.wqkv, 3 * d, d, true, 0.1 * @as(f32, @floatFromInt(li + ai + 1)));
            try fillMatrix(alloc, &a.wO, d, d, true, 0.2);
            try fillMatrix(alloc, &a.gate, 2 * cfg.mlp_hidden, d, true, 0.3);
            try fillVec(alloc, &l.attn_norm[ai], d, 1.5);
        }
        try fillMatrix(alloc, &l.w12, 2 * cfg.mlp_hidden, d, true, 0.4);
        try fillMatrix(alloc, &l.w3, d, cfg.mlp_hidden, true, 0.6);
        try fillVec(alloc, &l.mlp_norm, d, 1.75);
    }
    try fillVec(alloc, &m.out_norm, d, 2.5);
    try fillMatrix(alloc, &m.head.hidden, 2 * d, d, true, 0.8);
    try fillMatrix(alloc, &m.head.out, d, 2 * d, true, 0.9);
    try fillMatrix(alloc, &m.head.res, d, d, true, 0.05);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(std.testing.io, &buf);
    const path = try std.fs.path.join(testing.allocator, &.{ buf[0..n], "tiny_artifact.bin" });
    defer testing.allocator.free(path);

    const written = try save(&m, testing.allocator, std.testing.io, path);
    var back = try load(testing.allocator, std.testing.io, path);
    defer back.deinit();

    // the loader rejects trailing bytes, so reaching here means the byte count the
    // writer produced is exactly the byte count the canonical order predicts
    try testing.expect(written > 0);
    try testing.expectEqual(m.cfg.patch_size, back.cfg.patch_size);
    try testing.expectEqual(m.cfg.d_model, back.cfg.d_model);
    try testing.expectEqual(m.cfg.n_layers, back.cfg.n_layers);
    try testing.expectEqual(m.cfg.width_scale, back.cfg.width_scale);
    try testing.expectEqual(m.cfg.attend_missing, back.cfg.attend_missing);
    try testing.expectEqualSlices(model.Kind, cfg.schedule[0..cfg.n_layers], back.cfg.schedule[0..cfg.n_layers]);
    try testing.expectEqualSlices(f32, cfg.knots[0..cfg.n_knots], back.cfg.knots[0..cfg.n_knots]);

    try testing.expectEqualSlices(f32, m.patch.res.f32s, back.patch.res.f32s); // dense record
    try testing.expectEqualSlices(i8, m.patch.hidden.trits, back.patch.hidden.trits); // ternary record
    try testing.expectEqualSlices(f32, m.patch.hidden.scales, back.patch.hidden.scales);
    try testing.expectEqualSlices(f32, m.forecast_query, back.forecast_query);
    try testing.expectEqualSlices(f32, m.out_norm, back.out_norm);

    for (m.layers, back.layers, 0..) |a, b, li| {
        try testing.expectEqual(a.kind, b.kind);
        try testing.expectEqualSlices(f32, a.mlp_norm, b.mlp_norm);
        try testing.expectEqualSlices(i8, a.w12.trits, b.w12.trits);
        for (a.attn[0..a.kind.axes().len], b.attn[0..b.kind.axes().len], 0..) |x, y, ai| {
            try testing.expectEqualSlices(i8, x.wqkv.trits, y.wqkv.trits);
            try testing.expectEqualSlices(f32, x.wqkv.scales, y.wqkv.scales);
            try testing.expectEqualSlices(f32, a.attn_norm[ai], b.attn_norm[ai]);
            _ = li;
        }
    }
}
