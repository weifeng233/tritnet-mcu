//! Weight loading and ternary quantization for the Timestral example.
//!
//! `load` reads the released `model.safetensors` into the forward-pass structures;
//! `ternarize` applies the framework's absmean quantizer row by row. The eight layer
//! stack has 75 tensors with a fixed layout, so every matrix is fetched by name and
//! checked against its expected shape: a renamed or reshaped checkpoint fails loudly
//! instead of producing quietly wrong forecasts.

const std = @import("std");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");

pub const Error = error{
    BadShape,
    TensorNotFound,
    UnsupportedDtype,
    ShapeMismatch,
    Truncated,
    BadHeaderLength,
    BadHeader,
    OutOfMemory,
};

const MatSpec = struct { suffix: []const u8, rows: usize, cols: usize };

fn mat(buf: []u8, f: *safetensors.File, alloc: std.mem.Allocator, prefix: []const u8, suffix: []const u8, rows: usize, cols: usize) Error!model.Matrix {
    const name = std.fmt.bufPrint(buf, "{s}{s}", .{ prefix, suffix }) catch return error.BadShape;
    const w = f.f32s(alloc, name, &.{ rows, cols }) catch |e| return mapErr(e);
    return .{ .rows = rows, .cols = cols, .f32s = w };
}

fn bias(buf: []u8, f: *safetensors.File, alloc: std.mem.Allocator, prefix: []const u8, suffix: []const u8, n: usize) Error![]f32 {
    const name = std.fmt.bufPrint(buf, "{s}{s}", .{ prefix, suffix }) catch return error.BadShape;
    return f.f32s(alloc, name, &.{n}) catch |e| return mapErr(e);
}

fn mapErr(e: anyerror) Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.TensorNotFound => error.TensorNotFound,
        error.UnsupportedDtype => error.UnsupportedDtype,
        error.ShapeMismatch => error.ShapeMismatch,
        error.Truncated => error.Truncated,
        error.BadHeaderLength => error.BadHeaderLength,
        error.BadHeader => error.BadHeader,
        else => error.BadShape,
    };
}

/// Load the released checkpoint (fp32 weights, `model.safetensors`). The header is
/// parsed in a short-lived arena; every tensor lands in the model's own arena.
///
/// A missing or unreadable file keeps its own error name rather than being folded
/// into `Truncated`: "the checkpoint was never downloaded" and "the checkpoint is
/// corrupt" need different fixes, and the parity tests print whichever one happened.
pub fn load(backing: std.mem.Allocator, io: std.Io, path: []const u8) !model.Model {
    var scratch = std.heap.ArenaAllocator.init(backing);
    defer scratch.deinit();
    const file_bytes = std.Io.Dir.cwd().readFileAlloc(io, path, scratch.allocator(), .limited(1 << 30)) catch |err| switch (err) {
        error.StreamTooLong => return error.Truncated,
        else => |e| return e,
    };
    var f = try safetensors.File.openBytes(scratch.allocator(), file_bytes);
    var m = try model.Model.create(backing, model.config5m());
    errdefer m.deinit();
    const cfg = m.cfg;
    const d = cfg.d_model;
    const h = 4 * d; // ResidualBlock hidden width
    const npq = cfg.quantilesPerPatch();
    const alloc = m.arenaAlloc();
    var buf: [160]u8 = undefined;

    // patch embedding (the only block with biases)
    m.patch.hidden = try mat(&buf, &f, alloc, "representation.patch_proj.hidden_layer.", "weight", h, 2 * cfg.patch_size);
    m.patch.hidden_b = try bias(&buf, &f, alloc, "representation.patch_proj.hidden_layer.", "bias", h);
    m.patch.out = try mat(&buf, &f, alloc, "representation.patch_proj.output_layer.", "weight", d, h);
    m.patch.out_b = try bias(&buf, &f, alloc, "representation.patch_proj.output_layer.", "bias", d);
    m.patch.res = try mat(&buf, &f, alloc, "representation.patch_proj.residual_layer.", "weight", d, 2 * cfg.patch_size);
    m.patch.res_b = try bias(&buf, &f, alloc, "representation.patch_proj.residual_layer.", "bias", d);
    m.forecast_query = try f.f32s(alloc, "representation.forecast_query", &.{d});

    for (m.layers, 0..) |*layer, i| {
        layer.kind = cfg.schedule[i];
        var pre: [96]u8 = undefined;
        const p = std.fmt.bufPrint(&pre, "transformer.layers.{d}.", .{i}) catch return error.BadShape;
        for (layer.kind.axes(), 0..) |_, j| {
            var sub: [96]u8 = undefined;
            const ap = std.fmt.bufPrint(&sub, "{s}attns.{d}.", .{ p, j }) catch return error.BadShape;
            layer.attn[j].wqkv = try mat(&buf, &f, alloc, ap, "wQKV.weight", 3 * d, d);
            layer.attn[j].wO = try mat(&buf, &f, alloc, ap, "wO.weight", d, d);
            layer.attn[j].gate = try mat(&buf, &f, alloc, ap, "_gate.weight", d, d);
            var nb: [96]u8 = undefined;
            const np = std.fmt.bufPrint(&nb, "{s}attn_norms.{d}.", .{ p, j }) catch return error.BadShape;
            layer.attn_norm[j] = try bias(&buf, &f, alloc, np, "scale", d);
        }
        var mp: [96]u8 = undefined;
        const mpre = std.fmt.bufPrint(&mp, "{s}mlp.", .{p}) catch return error.BadShape;
        layer.w12 = try mat(&buf, &f, alloc, mpre, "w12.weight", 2 * cfg.mlp_hidden, d);
        layer.w3 = try mat(&buf, &f, alloc, mpre, "w3.weight", d, cfg.mlp_hidden);
        var nn: [96]u8 = undefined;
        const npre = std.fmt.bufPrint(&nn, "{s}mlp_norm.", .{p}) catch return error.BadShape;
        layer.mlp_norm = try bias(&buf, &f, alloc, npre, "scale", d);
    }
    m.out_norm = try f.f32s(alloc, "transformer.out_norm.scale", &.{d});
    m.head.hidden = try mat(&buf, &f, alloc, "head.projection.hidden_layer.", "weight", h, d);
    m.head.out = try mat(&buf, &f, alloc, "head.projection.output_layer.", "weight", npq, h);
    m.head.res = try mat(&buf, &f, alloc, "head.projection.residual_layer.", "weight", npq, d);
    return m;
}

pub const QuantStats = struct {
    ternary_params: usize = 0,
    fp32_params: usize = 0,
    neg: usize = 0,
    zero: usize = 0,
    pos: usize = 0,
    matrices: usize = 0,
};

/// Absmean row quantization with the framework's rule: `scale = mean(|row|)`,
/// `trit = +1` past `scale/2`, `-1` below `-scale/2`, else 0. One fp32 scale per
/// output row rides along in the packed artifact.
pub fn ternarizeMatrix(alloc: std.mem.Allocator, w: *model.Matrix, stats: *QuantStats) !void {
    if (w.tern) return;
    const trits = try alloc.alloc(i8, w.rows * w.cols);
    const scales = try alloc.alloc(f32, w.rows);
    for (0..w.rows) |r| {
        const row = w.f32s[r * w.cols ..][0..w.cols];
        var sum: f32 = 0;
        for (row) |v| sum += @abs(v);
        const scale = sum / @as(f32, @floatFromInt(w.cols));
        const t = 0.5 * scale;
        scales[r] = scale;
        const dst = trits[r * w.cols ..][0..w.cols];
        for (row, dst) |v, *o| {
            if (v > t) {
                o.* = 1;
                stats.pos += 1;
            } else if (v < -t) {
                o.* = -1;
                stats.neg += 1;
            } else {
                o.* = 0;
                stats.zero += 1;
            }
        }
        stats.ternary_params += w.cols;
    }
    w.trits = trits;
    w.scales = scales;
    w.tern = true;
    stats.matrices += 1;
}

/// Ternarize every projection (norms, the forecast query and the patch-embedding biases
/// stay fp32, exactly as the framework keeps embeddings and biases dense).
pub fn ternarizeModel(m: *model.Model, stats: *QuantStats) !void {
    const alloc = m.arenaAlloc();
    try ternarizeMatrix(alloc, &m.patch.hidden, stats);
    try ternarizeMatrix(alloc, &m.patch.out, stats);
    try ternarizeMatrix(alloc, &m.patch.res, stats);
    stats.fp32_params += m.patch.hidden_b.len + m.patch.out_b.len + m.patch.res_b.len;
    for (m.layers) |*l| {
        for (l.attn[0..l.kind.axes().len]) |*a| {
            try ternarizeMatrix(alloc, &a.wqkv, stats);
            try ternarizeMatrix(alloc, &a.wO, stats);
            try ternarizeMatrix(alloc, &a.gate, stats);
        }
        try ternarizeMatrix(alloc, &l.w12, stats);
        try ternarizeMatrix(alloc, &l.w3, stats);
        stats.fp32_params += l.attn_norm[0].len + l.attn_norm[1].len + l.mlp_norm.len;
    }
    try ternarizeMatrix(alloc, &m.head.hidden, stats);
    try ternarizeMatrix(alloc, &m.head.out, stats);
    try ternarizeMatrix(alloc, &m.head.res, stats);
    stats.fp32_params += m.out_norm.len + m.forecast_query.len;
}
