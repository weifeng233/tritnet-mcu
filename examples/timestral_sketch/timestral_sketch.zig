//! Timestral-5M deployment: quantize the released checkpoint with the framework's
//! ternary stack, score it, and serve the sketch-a-series forecast worker.
//!
//!     zig build example-timestral            # quantize + evaluate + write the artifact
//!     zig build timestral-quantize           # only the quantize leg
//!     zig build timestral-eval               # score fp32 vs ternary
//!     zig build timestral-serve              # the JSON-lines worker the web app drives
//!
//! The subcommands are also reachable directly:
//!
//!     timestral-cli quantize examples/timestral_sketch/modeldata/timestral-5m/model.safetensors examples/timestral_sketch/modeldata/timestral_packed.bin
//!     timestral-cli eval examples/timestral_sketch/modeldata/timestral_packed.bin --n 24 --h 96 --ctx 256
//!     timestral-cli serve examples/timestral_sketch/modeldata/timestral_packed.bin
//!
//! `serve` reads one `{"y": [...], "h": 96}` per line on stdin and answers
//! `{"q": [[q01..q99] per step], "ms": ..., "model": ...}` on stdout: one process, one
//! loaded artifact, no PyTorch and no per-request warm-up.

const std = @import("std");
const ts = @import("timestral");
const model = ts.model;
const weights = ts.weights;
const artifact = ts.artifact;
const engine = ts.engine;
const evalmod = ts.eval;
const fixture = ts.fixture;
const timing = @import("framework").timing;
// Build option (-Drequire-parity=true). Local runs may skip the parity tests when
// the 21 MB checkpoint has not been downloaded; CI sets this so a missing model is
// a red build instead of a green one that tested nothing.
const test_options = @import("test_options");

const clib = @cImport({
    @cInclude("stdio.h");
});

pub const SAFETENSORS = "examples/timestral_sketch/modeldata/timestral-5m/model.safetensors";
pub const ARTIFACT = "examples/timestral_sketch/modeldata/timestral_packed.bin";
pub const FIXTURE = "examples/timestral_sketch/modeldata/fixtures/timestral_fixture.bin";

fn say(comptime fmt: []const u8, args: anytype) void {
    var buf: [2048]u8 = undefined;
    const s = std.fmt.bufPrintZ(&buf, fmt, args) catch return;
    _ = clib.printf("%s", s.ptr);
    // the log lines are interleaved with the JSON-lines writer on stdout, and a pipe is
    // block-buffered by libc: flush so a parent process can read the ready line at once.
    _ = clib.fflush(null);
}

const Loaded = union(enum) {
    fp32: model.Model,
    tern: model.Model,

    fn deinit(self: *Loaded) void {
        switch (self.*) {
            .fp32 => |*m| m.deinit(),
            .tern => |*m| m.deinit(),
        }
    }

    fn ptr(self: *Loaded) *model.Model {
        return switch (self.*) {
            .fp32 => |*m| m,
            .tern => |*m| m,
        };
    }

    fn label(self: *const Loaded) []const u8 {
        return switch (self.*) {
            .fp32 => "fp32 safetensors",
            .tern => "ternary packed",
        };
    }
};

/// Sniff the file: packed artifacts start with "TRTL", everything else is a safetensors
/// checkpoint of the released fp32 weights.
fn loadAny(backing: std.mem.Allocator, io: std.Io, path: []const u8) !Loaded {
    const head = std.Io.Dir.cwd().readFileAlloc(io, path, backing, .limited(1 << 30)) catch return error.FileNotFound;
    defer backing.free(head);
    if (head.len >= 4 and std.mem.eql(u8, head[0..4], artifact.MAGIC)) {
        return .{ .tern = try artifact.load(backing, io, path) };
    }
    return .{ .fp32 = try weights.load(backing, io, path) };
}

// ---------------------------------------------------------------------------
// quantize
// ---------------------------------------------------------------------------

fn cmdQuantize(io: std.Io, gpa: std.mem.Allocator, in_path: []const u8, out_path: []const u8, quantize: bool, calibrate: bool) !void {
    var m = try weights.load(gpa, io, in_path);
    defer m.deinit();
    const params = m.paramCount();
    say("loaded {s}\n", .{in_path});
    say("  params {d}\n", .{params});
    say("  fp32 weights {d} B\n", .{params * 4});

    var stats = weights.QuantStats{};
    if (quantize) try weights.ternarizeModel(&m, &stats);
    const bytes = try artifact.save(&m, gpa, io, out_path);
    const total = stats.neg + stats.zero + stats.pos;
    if (quantize) {
        say("ternary quantization (absmean per row, trits 5-per-byte)\n", .{});
        say("  matrices {d} | trits -1 {d:.1}%  0 {d:.1}%  +1 {d:.1}%\n", .{
            stats.matrices,
            pct(stats.neg, total),
            pct(stats.zero, total),
            pct(stats.pos, total),
        });
        say("  ternary params {d} ({d:.1}% of {d}), fp32 params {d}\n", .{
            stats.ternary_params,
            pct(stats.ternary_params, params),
            params,
            stats.fp32_params,
        });
    } else {
        say("dense artifact: every projection kept fp32 (debug/parity mode)\n", .{});
    }
    say("  packed artifact {s}: {d} B ({d:.2}x smaller than fp32, {d:.2} bits/weight)\n", .{
        out_path,
        bytes,
        @as(f64, @floatFromInt(params * 4)) / @as(f64, @floatFromInt(bytes)),
        @as(f64, @floatFromInt(bytes * 8)) / @as(f64, @floatFromInt(params)),
    });

    if (!calibrate) return;
    // A ternarized quantile head is overconfident: the fan it emits is narrower than the
    // error it makes. One scale on the quantile increments, fitted on a calibration split
    // and carried in the artifact, restores the coverage.
    const gamma = try calibrateWidth(gpa, io, out_path);
    var eng = try engine.Engine.open(gpa, io, out_path);
    eng.model.cfg.width_scale = gamma;
    _ = try artifact.save(&eng.model, gpa, io, out_path);
    eng.deinit();
    say("  quantile width scale {d:.2} (calibrated on 96 strokes, seed 777)\n", .{gamma});
}

/// Grid-search the quantile-width scale that minimizes pinball on a calibration split.
fn calibrateWidth(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !f32 {
    var eng = try engine.Engine.open(gpa, io, path);
    defer eng.deinit();
    const cfg = eng.model.cfg;
    const ctx_len: usize = 256;
    const h: usize = 96;
    const n: usize = 96;
    const data = try evalmod.strokes(gpa, n, ctx_len + h, 777);
    defer gpa.free(data);
    const ctx = try gpa.alloc(f32, n * ctx_len);
    defer gpa.free(ctx);
    const truth = try gpa.alloc(f32, n * h);
    defer gpa.free(truth);
    for (0..n) |r| {
        @memcpy(ctx[r * ctx_len ..][0..ctx_len], data[r * (ctx_len + h) ..][0..ctx_len]);
        @memcpy(truth[r * h ..][0..h], data[r * (ctx_len + h) + ctx_len ..][0..h]);
    }
    // Two passes: score the grid, then pick the smallest scale that brings the 5-95 % band
    // up to ~90 % coverage without giving away more than 2 % of the pinball loss. Coverage
    // is what a fan chart promises, and the loss is flat between gamma 1.0 and 1.2.
    const grid = [_]f32{ 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2 };
    var pinball = [_]f64{0} ** grid.len;
    var cov = [_]f64{0} ** grid.len;
    var best: f64 = std.math.inf(f64);
    for (grid, 0..) |g, i| {
        eng.model.cfg.width_scale = g;
        var opts = engine.Options{};
        opts.symmetric = true;
        const q = try eng.forecast(gpa, ctx, n, ctx_len, h, opts);
        defer gpa.free(q);
        const score = evalmod.score(q, truth, ctx, n, ctx_len, h, cfg.knots[0..cfg.n_knots], cfg.medianIndex());
        pinball[i] = score.pinball;
        cov[i] = score.cov90;
        best = @min(best, score.pinball);
        say("    gamma {d:.2}: pinball {d:.5} cov90 {d:.3}\n", .{ g, score.pinball, score.cov90 });
    }
    var chosen: f32 = 1.0;
    for (grid, 0..) |g, i| {
        if (pinball[i] <= best * 1.02 and cov[i] >= 0.88) {
            chosen = g;
            break;
        }
    }
    return chosen;
}

fn pct(n: usize, d: usize) f64 {
    if (d == 0) return 0;
    return 100.0 * @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(d));
}

// ---------------------------------------------------------------------------
// eval
// ---------------------------------------------------------------------------

const EvalArgs = struct {
    n: usize = 24,
    ctx: usize = 256,
    h: usize = 96,
    seed: u64 = 2026,
    json: ?[]const u8 = null,
    dump: ?[]const u8 = null,
    batch: bool = true,
};

fn cmdEval(io: std.Io, gpa: std.mem.Allocator, path: []const u8, args: EvalArgs) !void {
    var loaded = try loadAny(gpa, io, path);
    var eng = engine.Engine.fromModel(loaded.ptr().*, gpa);
    defer eng.deinit();
    const cfg = eng.model.cfg;
    const params = eng.model.paramCount();
    const tern_params = eng.model.ternaryCount();
    const is_packed = std.meta.activeTag(loaded) == .tern;
    loaded = undefined; // ownership moved into the engine

    const data = try evalmod.strokes(gpa, args.n, args.ctx + args.h, args.seed);
    defer gpa.free(data);
    const ctx = try gpa.alloc(f32, args.n * args.ctx);
    defer gpa.free(ctx);
    const truth = try gpa.alloc(f32, args.n * args.h);
    defer gpa.free(truth);
    for (0..args.n) |r| {
        @memcpy(ctx[r * args.ctx ..][0..args.ctx], data[r * (args.ctx + args.h) ..][0..args.ctx]);
        @memcpy(truth[r * args.h ..][0..args.h], data[r * (args.ctx + args.h) + args.ctx ..][0..args.h]);
    }
    if (args.dump) |dp| {
        // raw f32 (strokes, ctx + h): the same corpus the Python side scores
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = dp, .data = std.mem.sliceAsBytes(data) });
        say("strokes  {s} ({d} x {d} f32)\n", .{ dp, args.n, args.ctx + args.h });
    }

    const t0 = timing.nowS();
    var quant: []f32 = undefined;
    if (args.batch) {
        var opts = engine.Options{};
        opts.symmetric = true;
        quant = try eng.forecast(gpa, ctx, args.n, args.ctx, args.h, opts);
    } else {
        quant = try gpa.alloc(f32, args.n * args.h * cfg.n_knots);
        for (0..args.n) |r| {
            var opts = engine.Options{};
            opts.symmetric = true;
            const one = try eng.forecast(gpa, ctx[r * args.ctx ..][0..args.ctx], 1, args.ctx, args.h, opts);
            @memcpy(quant[r * args.h * cfg.n_knots ..][0 .. args.h * cfg.n_knots], one);
            gpa.free(one);
        }
    }
    defer gpa.free(quant);
    const ms = (timing.nowS() - t0) * 1000.0;

    const stat = try std.Io.Dir.cwd().statFile(io, path, .{});
    const m = evalmod.score(quant, truth, ctx, args.n, args.ctx, args.h, cfg.knots[0..cfg.n_knots], cfg.medianIndex());
    say("model    {s}\n", .{path});
    say("params   {d} ({d} ternary, {d:.1}%)\n", .{ params, tern_params, pct(tern_params, params) });
    say("size     {d} B ({d:.2} bits/weight, {s})\n", .{
        stat.size,
        @as(f64, @floatFromInt(stat.size * 8)) / @as(f64, @floatFromInt(params)),
        if (is_packed) "packed artifact" else "fp32 safetensors",
    });
    say("task     {d} strokes x {d} ctx, horizon {d}, clampsym on\n", .{ args.n, args.ctx, args.h });
    say("pinball  {d:.5}\n", .{m.pinball});
    say("mae      {d:.5}\n", .{m.mae});
    say("mase     {d:.4}\n", .{m.mase});
    say("cov90    {d:.3}\n", .{m.cov90});
    say("beats-naive {d:.1}%\n", .{m.beat_naive * 100.0});
    say("latency  {d:.2} ms total, {d:.2} ms/stroke\n", .{ ms, ms / @as(f64, @floatFromInt(args.n)) });

    if (args.json) |jp| {
        var buf: [1024]u8 = undefined;
        const body = try std.fmt.bufPrint(&buf,
            \\{{"model":"{s}","params":{d},"ternary_params":{d},"bytes":{d},"bits_per_weight":{d:.4},
            \\"strokes":{d},"ctx":{d},"h":{d},"pinball":{d:.6},"mae":{d:.6},"mase":{d:.6},"cov90":{d:.4},
            \\"beats_naive":{d:.4},"ms_total":{d:.2},"ms_per_stroke":{d:.3}}}
        , .{
            path,                                                                     params, tern_params,                          stat.size,
            @as(f64, @floatFromInt(stat.size * 8)) / @as(f64, @floatFromInt(params)), args.n, args.ctx,                             args.h,
            m.pinball,                                                                m.mae,  m.mase,                               m.cov90,
            m.beat_naive,                                                             ms,     ms / @as(f64, @floatFromInt(args.n)),
        });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = jp, .data = body });
        say("report   {s}\n", .{jp});
    }
}

// ---------------------------------------------------------------------------
// forecast: batch inference without metrics, for the parity check
// ---------------------------------------------------------------------------

/// Read `n` strokes of `ctx + h` steps from a raw little-endian f32 file, forecast the
/// `h`-step continuation of each, and write the quantiles `(n, h, knots)` as raw f32.
/// `python/timestral_parity.py` feeds the same file to PyTorch and compares the two.
fn cmdForecast(io: std.Io, gpa: std.mem.Allocator, path: []const u8, corpus_path: []const u8, ctx: usize, h: usize, out_path: []const u8) !void {
    var loaded = try loadAny(gpa, io, path);
    var eng = engine.Engine.fromModel(loaded.ptr().*, gpa);
    defer eng.deinit();
    loaded = undefined;
    const cfg = eng.model.cfg;

    const raw = try std.Io.Dir.cwd().readFileAlloc(io, corpus_path, gpa, .limited(1 << 28));
    defer gpa.free(raw);
    if (raw.len % 4 != 0) return error.BadCorpus;
    const row = ctx + h;
    const floats = std.mem.bytesAsSlice(f32, raw);
    if (floats.len % row != 0) return error.BadCorpus;
    const n = floats.len / row;
    const ctx_slice = try gpa.alloc(f32, n * ctx);
    defer gpa.free(ctx_slice);
    for (0..n) |r| @memcpy(ctx_slice[r * ctx ..][0..ctx], floats[r * row ..][0..ctx]);

    var opts = engine.Options{};
    opts.symmetric = true;
    const q = try eng.forecast(gpa, ctx_slice, n, ctx, h, opts);
    defer gpa.free(q);
    // serialize as little-endian f32 regardless of host endianness
    const bytes = try gpa.alloc(u8, q.len * 4);
    defer gpa.free(bytes);
    for (q, 0..) |v, i| std.mem.writeInt(u32, bytes[4 * i ..][0..4], @bitCast(v), .little);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = bytes });
    say("forecast {d} strokes: {s} -> {s} ({d} x {d} x {d})\n", .{ n, corpus_path, out_path, n, h, cfg.n_knots });
}

// ---------------------------------------------------------------------------
// serve
// ---------------------------------------------------------------------------

fn cmdServe(io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
    var eng = try engine.Engine.open(gpa, io, path);
    defer eng.deinit();
    const cfg = eng.model.cfg;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    var in_buf: [1 << 18]u8 = undefined;
    var out_buf: [1 << 22]u8 = undefined;
    var reader = std.Io.File.stdin().readerStreaming(io, &in_buf);
    var writer = std.Io.File.stdout().writerStreaming(io, &out_buf);

    say("timestral worker ready: {s} ({d} params, {d} ternary)\n", .{ path, eng.model.paramCount(), eng.model.ternaryCount() });
    while (true) {
        _ = arena.reset(.retain_capacity);
        const work = arena.allocator();
        const line = reader.interface.takeDelimiter('\n') catch |e| {
            if (e == error.EndOfStream) break;
            return e;
        } orelse break;
        if (std.mem.trim(u8, line, " \r\t").len == 0) continue;

        const parsed = std.json.parseFromSlice(std.json.Value, work, line, .{}) catch {
            try reply(&writer, "{\"error\":\"bad json\"}");
            continue;
        };
        const obj = parsed.value.object;
        const y = (obj.get("y") orelse {
            try reply(&writer, "{\"error\":\"missing y\"}");
            continue;
        }).array.items;
        const h: usize = @intCast((obj.get("h") orelse {
            try reply(&writer, "{\"error\":\"missing h\"}");
            continue;
        }).integer);
        if (h == 0 or h > cfg.block_size * cfg.patch_size) {
            try reply(&writer, "{\"error\":\"bad h\"}");
            continue;
        }
        const vals = try work.alloc(f32, y.len);
        for (y, 0..) |v, i| {
            vals[i] = switch (v) {
                .float => |f| @floatCast(f),
                .integer => |n| @floatFromInt(n),
                else => std.math.nan(f32),
            };
        }
        const t0 = timing.nowS();
        var opts = engine.Options{};
        opts.symmetric = true;
        const q = try eng.forecast(work, vals, 1, y.len, h, opts);
        const ms = (timing.nowS() - t0) * 1000.0;
        try writeForecast(&writer, q, h, cfg.n_knots, ms);
    }
}

fn reply(w: anytype, body: []const u8) !void {
    try w.interface.writeAll(body);
    try w.interface.writeAll("\n");
    try w.interface.flush();
}

fn writeForecast(w: anytype, q: []const f32, h: usize, nq: usize, ms: f64) !void {
    try w.interface.print("{{\"q\":[", .{});
    for (0..h) |i| {
        if (i > 0) try w.interface.writeAll(",");
        try w.interface.writeAll("[");
        for (0..nq) |k| {
            if (k > 0) try w.interface.writeAll(",");
            const v = q[i * nq + k];
            if (std.math.isFinite(v)) {
                try w.interface.print("{d:.4}", .{v});
            } else {
                try w.interface.print("0", .{});
            }
        }
        try w.interface.writeAll("]");
    }
    try w.interface.print("],\"ms\":{d:.2},\"model\":\"Timestral-5M ternary\"}}\n", .{ms});
    try w.interface.flush();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

pub fn main(init: std.process.Init) u8 {
    const gpa = init.gpa;
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();

    var args = std.process.Args.Iterator.initAllocator(init.minimal.args, alloc) catch return 1;
    _ = args.next(); // argv[0]
    const cmd = args.next() orelse "help";

    if (std.mem.eql(u8, cmd, "quantize")) {
        const in_path = args.next() orelse SAFETENSORS;
        const out_path = args.next() orelse ARTIFACT;
        var quantize = true;
        var calibrate = true;
        while (args.next()) |a| {
            if (std.mem.eql(u8, a, "--fp32")) quantize = false;
            if (std.mem.eql(u8, a, "--no-calibrate")) calibrate = false;
        }
        cmdQuantize(io, gpa, in_path, out_path, quantize, quantize and calibrate) catch |e| {
            say("quantize failed: {s}\n", .{@errorName(e)});
            return 1;
        };
        return 0;
    }
    if (std.mem.eql(u8, cmd, "eval")) {
        const path = args.next() orelse ARTIFACT;
        var ea = EvalArgs{};
        while (args.next()) |a| {
            if (std.mem.eql(u8, a, "--n")) {
                ea.n = std.fmt.parseInt(usize, args.next() orelse "24", 10) catch 24;
            } else if (std.mem.eql(u8, a, "--ctx")) {
                ea.ctx = std.fmt.parseInt(usize, args.next() orelse "256", 10) catch 256;
            } else if (std.mem.eql(u8, a, "--h")) {
                ea.h = std.fmt.parseInt(usize, args.next() orelse "96", 10) catch 96;
            } else if (std.mem.eql(u8, a, "--seed")) {
                ea.seed = std.fmt.parseInt(u64, args.next() orelse "2026", 10) catch 2026;
            } else if (std.mem.eql(u8, a, "--json")) {
                ea.json = args.next();
            } else if (std.mem.eql(u8, a, "--dump")) {
                ea.dump = args.next();
            } else if (std.mem.eql(u8, a, "--serial")) {
                ea.batch = false;
            }
        }
        cmdEval(io, gpa, path, ea) catch |e| {
            say("eval failed: {s}\n", .{@errorName(e)});
            return 1;
        };
        return 0;
    }
    if (std.mem.eql(u8, cmd, "forecast")) {
        const model_path = args.next() orelse ARTIFACT;
        const corpus_path = args.next() orelse "examples/timestral_sketch/modeldata/timestral_strokes.bin";
        var ctx: usize = 256;
        var h: usize = 96;
        var out_path: []const u8 = "examples/timestral_sketch/modeldata/timestral_forecast.bin";
        while (args.next()) |a| {
            if (std.mem.eql(u8, a, "--ctx")) {
                ctx = std.fmt.parseInt(usize, args.next() orelse "256", 10) catch 256;
            } else if (std.mem.eql(u8, a, "--h")) {
                h = std.fmt.parseInt(usize, args.next() orelse "96", 10) catch 96;
            } else if (std.mem.eql(u8, a, "--out")) {
                out_path = args.next() orelse out_path;
            }
        }
        cmdForecast(io, gpa, model_path, corpus_path, ctx, h, out_path) catch |e| {
            say("forecast failed: {s}\n", .{@errorName(e)});
            return 1;
        };
        return 0;
    }
    if (std.mem.eql(u8, cmd, "serve")) {
        const path = args.next() orelse ARTIFACT;
        cmdServe(io, gpa, path) catch |e| {
            say("serve failed: {s}\n", .{@errorName(e)});
            return 1;
        };
        return 0;
    }

    say(
        \\usage: timestral-cli <command>
        \\  quantize [safetensors] [out.bin]   ternarize + pack the released checkpoint
        \\  eval <model> [--n N --ctx T --h H --json out]   score pinball/mase
        \\  serve <artifact.bin>                JSON-lines forecast worker (stdin/stdout)
        \\
    , .{});
    return 0;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Compare one fixture tensor against what the Zig engine produced at the same stage.
const Comparator = struct {
    fx: *const fixture.Fixture,
    worst: f32 = 0,
    fails: usize = 0,
    compared: usize = 0,
    amp: f32 = 1,

    fn hook(self: *Comparator, stage: model.Stage, layer: usize, data: []const f32) void {
        var buf: [96]u8 = undefined;
        const name = switch (stage) {
            .scaled => "scaled",
            .x_in => "x_in",
            .out_norm => "out_norm",
            .head_raw => "head_out",
            .block_quantiles => "block_quantiles",
            .layer => std.fmt.bufPrint(&buf, "layer{d}.out", .{layer}) catch return,
        };
        const t = self.fx.get(name) catch return;
        self.compared += 1;
        if (t.data.len != data.len) {
            std.debug.print("stage {s}: length {d} != fixture {d}\n", .{ name, data.len, t.data.len });
            self.fails += 1;
            return;
        }
        var max_abs: f32 = 0;
        var diff: f32 = 0;
        for (data, t.data) |a, b| {
            max_abs = @max(max_abs, @abs(b));
            diff = @max(diff, @abs(a - b));
        }
        self.worst = @max(self.worst, diff / @max(max_abs, 1e-6));
        if (diff > 1e-4 * @max(max_abs, self.amp)) {
            std.debug.print("stage {s}: max |diff| {e:.3} on scale {e:.3}\n", .{ name, diff, max_abs });
            self.fails += 1;
        }
    }

    fn callback(ctx: *anyopaque, stage: model.Stage, layer: usize, data: []const f32) void {
        const self: *Comparator = @ptrCast(@alignCast(ctx));
        self.hook(stage, layer, data);
    }
};

/// Turn "the model is not on this machine" into a skip -- except under
/// -Drequire-parity=true, where it is a failure. A parity test that silently
/// skips in CI is worse than no parity test: it reports success for work it
/// did not do.
fn skipUnlessRequired(err: anyerror, comptime what: []const u8, path: []const u8) anyerror {
    if (test_options.require_parity) return err;
    std.debug.print("skip parity: {s} at {s} unavailable ({s})\n", .{ what, path, @errorName(err) });
    return error.SkipZigTest;
}

test "zig decode block matches the reference fixtures" {
    const gpa = testing.allocator;
    var fx = fixture.Fixture.load(gpa, std.testing.io, FIXTURE) catch |err| return skipUnlessRequired(err, "fixture", FIXTURE);
    defer fx.deinit();
    var m = weights.load(gpa, std.testing.io, SAFETENSORS) catch |err| return skipUnlessRequired(err, "safetensors", SAFETENSORS);
    defer m.deinit();

    const inputs_t = try fx.get("inputs");
    const observed_t = try fx.get("observed");
    const rows = inputs_t.dims[0];
    const t = inputs_t.dims[1];
    const expected = try fx.get("block_quantiles");
    const h_patches = 8;
    const npq = m.cfg.quantilesPerPatch();

    const observed: []u8 = try gpa.alloc(u8, rows * t);
    defer gpa.free(observed);
    for (observed, observed_t.data) |*o, v| o.* = if (v != 0) 1 else 0;
    const out = try gpa.alloc(f32, rows * h_patches * npq);
    defer gpa.free(out);
    const loc = try gpa.alloc(f32, rows);
    defer gpa.free(loc);
    const scale = try gpa.alloc(f32, rows);
    defer gpa.free(scale);

    var cmp = Comparator{ .fx = &fx };
    _ = &cmp;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    try m.decodeBlockHooked(arena.allocator(), inputs_t.data, observed, rows, t / m.cfg.patch_size, h_patches, out, loc, scale, .{
        .ctx = &cmp,
        .f = Comparator.callback,
    });
    std.debug.print("parity: {d} stages, worst relative diff {e:.2}\n", .{ cmp.compared, cmp.worst });
    try testing.expect(cmp.compared >= 12);
    try testing.expectEqual(@as(usize, 0), cmp.fails);
    try testing.expect(cmp.worst < 1e-4);
    try testing.expectEqual(expected.data.len, out.len);
}

test "engine forecast (padding, clampsym, refine, cap) matches the reference" {
    const gpa = testing.allocator;
    var fx = fixture.Fixture.load(gpa, std.testing.io, FIXTURE) catch |err| return skipUnlessRequired(err, "fixture", FIXTURE);
    defer fx.deinit();
    const inputs_t = try fx.get("inputs");
    const expected = try fx.get("forecast_sym"); // fc.forecast(..., symmetric_averaging=True)
    const rows = inputs_t.dims[0];
    const ctx = inputs_t.dims[1];

    var m = weights.load(gpa, std.testing.io, SAFETENSORS) catch |err| return skipUnlessRequired(err, "safetensors", SAFETENSORS);
    var eng = engine.Engine.fromModel(m, gpa);
    defer eng.deinit();
    m = undefined;
    // fixture shape is (batch, variates, horizon, quantiles); this app is univariate
    try testing.expectEqual(@as(usize, 4), expected.dims.len);
    try testing.expectEqual(@as(usize, 1), expected.dims[1]);
    const h = expected.dims[2];
    const nq = expected.dims[3];

    // the fixture stroke is fully observed (the mask was baked into zeros before the
    // reference forecast), so the engine's own NaN handling is not exercised here
    var opts = engine.Options{};
    opts.symmetric = true;
    const q = try eng.forecast(gpa, inputs_t.data, rows, ctx, h, opts);
    defer gpa.free(q);
    try testing.expectEqual(rows * h * nq, q.len);

    var diff: f32 = 0;
    var mag: f32 = 0;
    for (q, expected.data) |a, b| {
        diff = @max(diff, @abs(a - b));
        mag = @max(mag, @abs(b));
    }
    std.debug.print("engine parity: max |diff| {e:.3} on magnitude {e:.3}\n", .{ diff, mag });
    try testing.expect(diff / mag < 1e-3);
}

test "packed artifact round-trips trits, scales and fp32 tensors" {
    const gpa = testing.allocator;
    var m = weights.load(gpa, std.testing.io, SAFETENSORS) catch |err| return skipUnlessRequired(err, "safetensors", SAFETENSORS);
    defer m.deinit();
    var stats = weights.QuantStats{};
    try weights.ternarizeModel(&m, &stats);
    // Written into a test temp dir, not zig-out/: `zig build test` has to pass in a
    // clean clone, where zig-out/ does not exist yet.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(std.testing.io, &buf);
    const path = try std.fs.path.join(gpa, &.{ buf[0..n], "timestral_roundtrip.bin" });
    defer gpa.free(path);
    const written = try artifact.save(&m, gpa, std.testing.io, path);
    try testing.expect(written > 0);
    var back = try artifact.load(gpa, std.testing.io, path);
    defer back.deinit();
    try testing.expectEqual(m.cfg.d_model, back.cfg.d_model);
    try testing.expectEqual(m.cfg.n_layers, back.cfg.n_layers);
    // spot-check the first ternary row and the fp32 forecast query
    try testing.expectEqualSlices(i8, m.layers[0].attn[0].wqkv.trits[0..512], back.layers[0].attn[0].wqkv.trits[0..512]);
    try testing.expectEqualSlices(f32, m.layers[0].attn[0].wqkv.scales, back.layers[0].attn[0].wqkv.scales);
    try testing.expectEqualSlices(f32, m.forecast_query, back.forecast_query);
}
