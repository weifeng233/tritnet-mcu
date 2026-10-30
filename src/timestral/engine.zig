//! The forecast engine: request-shaped decoding on top of `model.decodeBlock`.
//!
//! A stroke becomes a left-padded context, the two-pass clampsym request (the stroke and
//! its negation, the mirrored forecast reflected back and averaged) is one batched decode,
//! and the block then moves from the anchor frame to the advancing refine frame it was
//! trained in — the same sequence the reference forecaster runs. Output is `(rows, h, Q)`
//! raw-space quantiles.

const std = @import("std");
const model = @import("model.zig");
const artifact = @import("artifact.zig");

pub const Options = struct {
    /// Average the forecast with the reflected forecast of `-stroke` (the reference's
    /// symmetric averaging; the Space enables it for every request).
    symmetric: bool = true,
    /// Clamp to `[min - k*scale, max + k*scale]` of the observed context; 0 disables.
    cap_k: f32 = 1e4,
    /// Clamp at 0 the strokes whose observed context is entirely non-negative.
    clip_nonnegative: bool = false,
};

pub const Engine = struct {
    model: model.Model,
    arena: std.heap.ArenaAllocator,
    path: []const u8 = "",

    pub fn open(backing: std.mem.Allocator, io: std.Io, path: []const u8) !Engine {
        const m = try artifact.load(backing, io, path);
        var eng = Engine.fromModel(m, backing);
        eng.path = eng.arena.allocator().dupe(u8, path) catch path;
        return eng;
    }

    /// Wrap an already-loaded model (the fp32 checkpoint, for A/B scoring); the engine
    /// takes ownership and frees it in `deinit`.
    pub fn fromModel(m: model.Model, backing: std.mem.Allocator) Engine {
        return .{
            .model = m,
            .arena = std.heap.ArenaAllocator.init(backing),
        };
    }

    pub fn deinit(self: *Engine) void {
        self.model.deinit();
        self.arena.deinit();
    }

    pub fn allocator(self: *Engine) std.mem.Allocator {
        return self.arena.allocator();
    }

    /// Forecast `h` steps after each of `rows` strokes of `t` steps (`t` may be anything;
    /// the context is left-padded to a patch multiple, as the reference does).
    /// Returns `(rows, h, n_knots)` raw-space quantiles owned by the caller's allocator.
    pub fn forecast(
        self: *Engine,
        alloc: std.mem.Allocator,
        ctx: []const f32,
        rows: usize,
        t: usize,
        h: usize,
        opts: Options,
    ) ![]f32 {
        const cfg = self.model.cfg;
        const ps = cfg.patch_size;
        if (h == 0 or h > cfg.block_size * ps) return error.BadHorizon;
        if (ctx.len != rows * t) return error.BadShape;
        _ = self.arena.reset(.retain_capacity);
        const work = self.arena.allocator();

        const tpad = ((t + ps - 1) / ps) * ps;
        const pad = tpad - t;
        const passes: usize = if (opts.symmetric) 2 else 1;
        const rows2 = rows * passes;
        const h_patches = (h + ps - 1) / ps;

        const inputs = try work.alloc(f32, rows2 * tpad);
        const observed = try work.alloc(u8, rows2 * tpad);
        @memset(inputs, 0);
        @memset(observed, 0);
        for (0..rows) |r| {
            for (0..t) |i| {
                const v = ctx[r * t + i];
                const dst = r * tpad + pad + i;
                if (std.math.isNan(v)) {
                    inputs[dst] = 0;
                    observed[dst] = 0;
                } else {
                    inputs[dst] = v;
                    observed[dst] = 1;
                }
            }
        }
        if (opts.symmetric) {
            for (0..rows) |r| {
                for (0..tpad) |i| {
                    const src = r * tpad + i;
                    const dst = (rows + r) * tpad + i;
                    inputs[dst] = -inputs[src];
                    observed[dst] = observed[src];
                }
            }
        }

        const npq = cfg.quantilesPerPatch();
        const quant = try work.alloc(f32, rows2 * h_patches * npq);
        const anchor_loc = try work.alloc(f32, rows2);
        const anchor_scale = try work.alloc(f32, rows2);
        try self.model.decodeBlock(work, inputs, observed, rows2, tpad / ps, h_patches, quant, anchor_loc, anchor_scale);

        try refineBlock(work, inputs, observed, rows2, tpad, ps, h_patches, cfg.min_scale, anchor_loc, anchor_scale, quant, cfg.n_knots, cfg.medianIndex());
        if (opts.cap_k > 0) try capBlock(work, inputs, observed, rows2, tpad, anchor_scale, opts.cap_k, quant, cfg.n_knots);

        const out = try alloc.alloc(f32, rows * h * cfg.n_knots);
        const step_span = h_patches * npq;
        for (0..rows) |r| {
            for (0..h) |i| {
                for (0..cfg.n_knots) |k| {
                    const a = quant[r * step_span + i * cfg.n_knots + k];
                    var v = a;
                    if (opts.symmetric) {
                        const b = quant[(rows + r) * step_span + i * cfg.n_knots + (cfg.n_knots - 1 - k)];
                        v = 0.5 * (a - b);
                    }
                    out[(r * h + i) * cfg.n_knots + k] = v;
                }
            }
        }
        if (opts.clip_nonnegative) {
            for (0..rows) |r| {
                var any = false;
                var all_nonneg = true;
                for (0..t) |i| {
                    const v = ctx[r * t + i];
                    if (!std.math.isNan(v)) {
                        any = true;
                        if (v < 0) all_nonneg = false;
                    }
                }
                if (any and all_nonneg) {
                    for (out[r * h * cfg.n_knots ..][0 .. h * cfg.n_knots]) |*v| v.* = @max(v.*, 0);
                }
            }
        }
        return out;
    }
};

/// Welford state merge, as in the reference scaler.
fn welfordCombine(w_a: f64, mean_a: f64, m2_a: f64, w_b: f64, mean_b: f64, m2_b: f64, out: *[3]f64) void {
    const w = w_a + w_b;
    const w_safe = @max(w, std.math.floatMin(f64));
    const delta = mean_b - mean_a;
    var mean = mean_a + delta * w_b / w_safe;
    var m2 = m2_a + m2_b + delta * delta * w_a * w_b / w_safe;
    if (w == 0) {
        mean = 0;
        m2 = 0;
    }
    out.* = .{ w, mean, m2 };
}

/// Move one decoded block into the advancing refine frame: each horizon patch j is
/// normalized with the context statistics plus the block's medians of patches 0..j-1.
fn refineBlock(
    alloc: std.mem.Allocator,
    inputs: []const f32,
    observed: []const u8,
    rows: usize,
    t: usize,
    ps: usize,
    h_patches: usize,
    min_scale: f32,
    anchor_loc: []const f32,
    anchor_scale: []const f32,
    quant: []f32,
    n_knots: usize,
    median_idx: usize,
) !void {
    if (h_patches <= 1) return;
    const patches = t / ps;
    const scan = try alloc.alloc([3]f64, patches);
    const locs = try alloc.alloc(f64, h_patches);
    const scales = try alloc.alloc(f64, h_patches);
    const medians = try alloc.alloc(f64, ps);
    const folded = try alloc.alloc(f64, ps);
    const clip: f64 = @sqrt(@as(f64, std.math.floatMax(f32))) / 1e3;

    for (0..rows) |r| {
        // per-patch moments of the context, then an inclusive Welford scan
        for (0..patches) |p| {
            var cnt: f64 = 0;
            var sum: f64 = 0;
            for (0..ps) |i| {
                const idx = r * t + p * ps + i;
                if (observed[idx] != 0) {
                    cnt += 1;
                    sum += inputs[idx];
                }
            }
            const mean = if (cnt > 0) sum / cnt else 0;
            var m2: f64 = 0;
            for (0..ps) |i| {
                const idx = r * t + p * ps + i;
                if (observed[idx] != 0) m2 += (inputs[idx] - mean) * (inputs[idx] - mean);
            }
            scan[p] = .{ cnt, mean, m2 };
        }
        var step: usize = 1;
        while (step < patches) : (step *= 2) {
            var i = patches;
            while (i > step) {
                i -= 1;
                var merged: [3]f64 = undefined;
                welfordCombine(scan[i - step][0], scan[i - step][1], scan[i - step][2], scan[i][0], scan[i][1], scan[i][2], &merged);
                scan[i] = merged;
            }
        }
        var cnt = scan[patches - 1][0];
        var mean = scan[patches - 1][1];
        var m2 = scan[patches - 1][2];
        const offset = mean;
        const span = h_patches * ps * n_knots;
        for (0..h_patches) |j| {
            const variance = @max(m2 / @max(cnt - 1.0, 1.0), 0.0);
            const scale_j = @max(@sqrt(variance), @as(f64, min_scale));
            locs[j] = mean;
            scales[j] = scale_j;
            for (0..ps) |i| {
                const k = (j * ps + i) * n_knots + median_idx;
                const m = quant[r * span + k];
                medians[i] = (@as(f64, m) - @as(f64, anchor_loc[r])) / @as(f64, anchor_scale[r]);
                const v = medians[i] * scale_j + mean - offset;
                folded[i] = std.math.clamp(v, -clip, clip) + offset;
            }
            var b_mean: f64 = 0;
            for (folded) |v| b_mean += v;
            b_mean /= @as(f64, @floatFromInt(ps));
            var b_m2: f64 = 0;
            for (folded) |v| b_m2 += (v - b_mean) * (v - b_mean);
            var merged: [3]f64 = undefined;
            welfordCombine(cnt, mean, m2, @floatFromInt(ps), b_mean, b_m2, &merged);
            cnt = merged[0];
            mean = merged[1];
            m2 = merged[2];
        }

        // re-denormalize every step of the block in its own frame
        for (0..h_patches * ps) |i| {
            const j = i / ps;
            const loc_j = if (j == 0) @as(f64, anchor_loc[r]) else locs[j];
            const scale_j = if (j == 0) @as(f64, anchor_scale[r]) else scales[j];
            const base = r * span + i * n_knots;
            const vals = quant[base..][0..n_knots];
            for (vals) |*v| {
                const normed = (@as(f64, v.*) - @as(f64, anchor_loc[r])) / @as(f64, anchor_scale[r]);
                v.* = @floatCast(normed * scale_j + loc_j);
            }
            clampStep(vals);
        }
    }
}

/// Clamp to `[min - k*scale, max + k*scale]` of each row's observed context.
fn capBlock(
    alloc: std.mem.Allocator,
    inputs: []const f32,
    observed: []const u8,
    rows: usize,
    t: usize,
    anchor_scale: []const f32,
    k: f32,
    quant: []f32,
    n_knots: usize,
) !void {
    _ = alloc;
    const span = quant.len / rows;
    for (0..rows) |r| {
        var lo: f32 = 0;
        var hi: f32 = 0;
        var any = false;
        var mn: f32 = std.math.floatMax(f32);
        var mx: f32 = -std.math.floatMax(f32);
        for (0..t) |i| {
            const idx = r * t + i;
            if (observed[idx] == 0) continue;
            const v = inputs[idx];
            if (std.math.isNan(v)) continue;
            any = true;
            mn = @min(mn, v);
            mx = @max(mx, v);
        }
        if (any) {
            lo = mn;
            hi = mx;
        }
        const pad = anchor_scale[r] * k;
        const lower = lo - pad;
        const upper = hi + pad;
        for (quant[r * span ..][0..span]) |*v| {
            if (std.math.isNan(v.*)) continue;
            v.* = std.math.clamp(v.*, lower, upper);
        }
    }
    _ = n_knots;
}

/// `clamp_nonfinite` on one step's quantile vector: infinities become the nearest finite
/// value on the axis, and a wholly non-finite axis becomes zero.
pub fn clampStep(vals: []f32) void {
    var max_finite: f32 = -std.math.floatMax(f32);
    var min_finite: f32 = std.math.floatMax(f32);
    var any_finite = false;
    for (vals) |v| {
        if (std.math.isFinite(v)) {
            any_finite = true;
            max_finite = @max(max_finite, v);
            min_finite = @min(min_finite, v);
        }
    }
    if (!any_finite) {
        @memset(vals, 0);
        return;
    }
    for (vals) |*v| {
        if (v.* == std.math.inf(f32)) {
            v.* = max_finite;
        } else if (v.* == -std.math.inf(f32)) {
            v.* = min_finite;
        }
    }
}
