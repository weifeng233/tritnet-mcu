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
    return m;
}
