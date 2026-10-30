//! Timestral inference in Zig: the deploy leg of the sketch-a-series example.
//!
//! The network is the released patch-based transformer: per-patch causal normalization
//! with an asinh squash, patch embedding from `[values, missing]`, `[Enc ; Dec]` tokens
//! through time (and variate) attention with RoPE and sigmoid gates, a SwiGLU MLP per
//! layer, and a monotone quantile head whose output is denormalized in the last context
//! patch's frame.
//!
//! Only the parts the app needs are implemented: one variate per series (so variate
//! attention degenerates to its value path, exactly what the reference computes when
//! every row is its own group), no covariates, and one decode block of up to 32 patches.
//! The block-autoregressive loop and the advancing refine frame live in engine.zig.
//!
//! Weights are either fp32 (`Matrix.f32s`) or ternary (`Matrix.trits` plus one absmean
//! scale per output row, `Matrix.scales`). The ternary path multiplies by -1/0/+1 only:
//! accumulate add/skip/subtract over the row, then scale once, like the C engines in `c/`.

const std = @import("std");

pub const MAX_LAYERS = 16;
pub const MAX_KNOTS = 32;
pub const MAX_HEAD_DIM = 256;

/// Stages of one decode pass, in order. The parity test compares each of them against
/// the PyTorch reference's dump, so a mismatch names the stage that diverged.
pub const Stage = enum {
    scaled,
    x_in,
    layer,
    out_norm,
    head_raw,
    block_quantiles,
};

pub const Hook = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, stage: Stage, layer: usize, data: []const f32) void,

    pub fn call(self: Hook, stage: Stage, layer: usize, data: []const f32) void {
        self.f(self.ctx, stage, layer, data);
    }
};

fn emit(hook: ?Hook, stage: Stage, layer: usize, data: []const f32) void {
    if (hook) |h| h.call(stage, layer, data);
}

pub const Kind = enum(u8) {
    time = 0,
    variate = 1,
    time_variate = 2,
    variate_time = 3,

    pub fn axes(self: Kind) []const bool {
        return switch (self) {
            .time => &.{true},
            .variate => &.{false},
            .time_variate => &.{ true, false },
            .variate_time => &.{ false, true },
        };
    }
};

pub const Config = struct {
    patch_size: usize = 32,
    d_model: usize = 192,
    num_heads: usize = 3,
    mlp_hidden: usize = 512,
    n_layers: usize = 8,
    schedule: [MAX_LAYERS]Kind = [_]Kind{.time} ** MAX_LAYERS,
    n_knots: usize = 21,
    knots: [MAX_KNOTS]f32 = [_]f32{0.0} ** MAX_KNOTS,
    width_factor: f32 = 3.0,
    /// Post-quantization quantile-width calibration (1.0 = the trained widths). Fitted on
    /// a calibration split after ternarization, because a ternarized quantile head is
    /// systematically overconfident: the fan is too narrow for the error it makes.
    width_scale: f32 = 1.0,
    norm_eps: f32 = 1e-6,
    rope_dim: usize = 32,
    rope_theta: f32 = 10000.0,
    min_scale: f32 = 1e-6,
    block_size: usize = 32,
    use_asinh: bool = true,
    attend_missing: bool = true,

    pub fn headDim(self: Config) usize {
        return self.d_model / self.num_heads;
    }

    pub fn medianIndex(self: Config) usize {
        for (self.knots[0..self.n_knots], 0..) |k, i| {
            if (k == 0.5) return i;
        }
        return self.n_knots / 2;
    }

    pub fn quantilesPerPatch(self: Config) usize {
        return self.patch_size * self.n_knots;
    }
};

/// The released Timestral-5M configuration (`config.json`).
pub fn config5m() Config {
    var c = Config{};
    const schedule = [_]Kind{ .time, .time, .time, .time_variate, .time, .time, .time, .time_variate };
    c.schedule = [_]Kind{.time} ** MAX_LAYERS;
    for (schedule, 0..) |k, i| c.schedule[i] = k;
    c.n_layers = schedule.len;
    const knots = [_]f32{ 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99 };
    c.knots = [_]f32{0.0} ** MAX_KNOTS;
    for (knots, 0..) |k, i| c.knots[i] = k;
    c.n_knots = knots.len;
    return c;
}

pub const Matrix = struct {
    rows: usize = 0,
    cols: usize = 0,
    tern: bool = false,
    f32s: []f32 = &.{},
    trits: []i8 = &.{},
    scales: []f32 = &.{},

    /// `y = W x`, vectorized 8 lanes at a time. The ternary path multiplies the input
    /// by the trit (+1/0/-1) and scales the finished row once: no per-weight table.
    pub fn gemv(self: *const Matrix, x: []const f32, y: []f32) void {
        std.debug.assert(x.len >= self.cols);
        std.debug.assert(y.len >= self.rows);
        const L = 8;
        const V = @Vector(L, f32);
        const T = @Vector(L, i8);
        const xs = x[0..self.cols];
        if (self.tern) {
            for (0..self.rows) |r| {
                const row = self.trits[r * self.cols ..][0..self.cols];
                var acc: V = @splat(0);
                var j: usize = 0;
                while (j + L <= self.cols) : (j += L) {
                    const t: T = row[j..][0..L].*;
                    const xv: V = xs[j..][0..L].*;
                    acc += @as(V, @floatFromInt(t)) * xv;
                }
                var tail: f32 = 0;
                while (j < self.cols) : (j += 1) tail += xs[j] * @as(f32, @floatFromInt(row[j]));
                y[r] = (@reduce(.Add, acc) + tail) * self.scales[r];
            }
        } else {
            for (0..self.rows) |r| {
                const row = self.f32s[r * self.cols ..][0..self.cols];
                var acc: V = @splat(0);
                var j: usize = 0;
                while (j + L <= self.cols) : (j += L) {
                    const wv: V = row[j..][0..L].*;
                    const xv: V = xs[j..][0..L].*;
                    acc += wv * xv;
                }
                var tail: f32 = 0;
                while (j < self.cols) : (j += 1) tail += row[j] * xs[j];
                y[r] = @reduce(.Add, acc) + tail;
            }
        }
    }

    /// A contiguous range of output rows, as its own matrix.
    pub fn rowView(self: *const Matrix, start: usize, count: usize) Matrix {
        if (self.tern) {
            return .{
                .rows = count,
                .cols = self.cols,
                .tern = true,
                .trits = self.trits[start * self.cols ..][0 .. count * self.cols],
                .scales = self.scales[start..][0..count],
            };
        }
        return .{
            .rows = count,
            .cols = self.cols,
            .f32s = self.f32s[start * self.cols ..][0 .. count * self.cols],
        };
    }

    pub fn params(self: *const Matrix) usize {
        return self.rows * self.cols;
    }
};

pub const Residual = struct {
    hidden: Matrix = .{},
    hidden_b: []f32 = &.{},
    out: Matrix = .{},
    out_b: []f32 = &.{},
    res: Matrix = .{},
    res_b: []f32 = &.{},

    /// `out(silu(hidden(x))) + res(x)` into `y` (`out.rows` long); `scratch` holds `hidden.rows`.
    pub fn forward(self: *const Residual, x: []const f32, scratch: []f32, y: []f32) void {
        self.hidden.gemv(x, scratch);
        for (scratch[0..self.hidden.rows], 0..) |*v, i| {
            if (self.hidden_b.len > 0) v.* += self.hidden_b[i];
            v.* = silu(v.*);
        }
        self.out.gemv(scratch, y);
        for (y[0..self.out.rows], 0..) |*v, i| {
            if (self.out_b.len > 0) v.* += self.out_b[i];
        }
        self.res.gemv(x, scratch[0..self.res.rows]);
        for (y[0..self.res.rows], 0..) |*v, i| {
            v.* += scratch[i];
            if (self.res_b.len > 0) v.* += self.res_b[i];
        }
    }
};

pub const Attn = struct {
    wqkv: Matrix = .{},
    wO: Matrix = .{},
    gate: Matrix = .{},

    pub fn hasGate(self: *const Attn) bool {
        return self.gate.rows > 0;
    }
};

pub const Layer = struct {
    kind: Kind = .time,
    attn: [2]Attn = .{ .{}, .{} },
    attn_norm: [2][]f32 = .{ &.{}, &.{} },
    mlp_norm: []f32 = &.{},
    w12: Matrix = .{},
    w3: Matrix = .{},
};

pub const Model = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config = .{},
    patch: Residual = .{},
    forecast_query: []f32 = &.{},
    layers: []Layer = &.{},
    out_norm: []f32 = &.{},
    head: Residual = .{},

    pub fn create(backing: std.mem.Allocator, cfg: Config) !Model {
        var arena = std.heap.ArenaAllocator.init(backing);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const layers = try alloc.alloc(Layer, cfg.n_layers);
        @memset(layers, Layer{});
        // the layer kind decides how many attention axes run, and therefore the record
        // order of a packed artifact: set it once here so every loader agrees
        for (layers, 0..) |*l, i| l.kind = cfg.schedule[i];
        return .{
            .arena = arena,
            .cfg = cfg,
            .layers = layers,
        };
    }

    pub fn deinit(self: *Model) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn arenaAlloc(self: *Model) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn paramCount(self: *const Model) usize {
        var n: usize = self.forecast_query.len;
        n += residualParams(&self.patch);
        for (self.layers) |*l| {
            for (l.attn[0..l.kind.axes().len]) |*a| {
                n += a.wqkv.params() + a.wO.params() + a.gate.params();
            }
            n += l.w12.params() + l.w3.params();
        }
        n += self.out_norm.len;
        n += residualParams(&self.head);
        return n;
    }

    pub fn ternaryCount(self: *const Model) usize {
        var n: usize = 0;
        n += residualTernary(&self.patch);
        for (self.layers) |*l| {
            for (l.attn[0..l.kind.axes().len]) |*a| {
                if (a.wqkv.tern) n += a.wqkv.params();
                if (a.wO.tern) n += a.wO.params();
                if (a.gate.tern) n += a.gate.params();
            }
            if (l.w12.tern) n += l.w12.params();
            if (l.w3.tern) n += l.w3.params();
        }
        n += residualTernary(&self.head);
        return n;
    }

    /// One decode block: `rows` independent univariate series with `patches` context
    /// patches, decoding `h_patches` forecast patches. Writes raw-space quantiles
    /// `(rows, h_patches * patch_size, n_knots)` and each row's anchor frame.
    pub fn decodeBlock(
        self: *const Model,
        alloc: std.mem.Allocator,
        inputs: []const f32,
        observed: []const u8,
        rows: usize,
        patches: usize,
        h_patches: usize,
        out: []f32,
        anchor_loc: []f32,
        anchor_scale: []f32,
    ) !void {
        return self.decodeBlockHooked(alloc, inputs, observed, rows, patches, h_patches, out, anchor_loc, anchor_scale, null);
    }

    /// `decodeBlock` with a per-stage hook; the parity test walks the same stages the
    /// PyTorch reference dumps, so a mismatch names the stage instead of the whole pass.
    pub fn decodeBlockHooked(
        self: *const Model,
        alloc: std.mem.Allocator,
        inputs: []const f32,
        observed: []const u8,
        rows: usize,
        patches: usize,
        h_patches: usize,
        out: []f32,
        anchor_loc: []f32,
        anchor_scale: []f32,
        hook: ?Hook,
    ) !void {
        const cfg = self.cfg;
        const ps = cfg.patch_size;
        const d = cfg.d_model;
        const hd = cfg.headDim();
        const s = patches + h_patches;
        const t = patches * ps;
        if (inputs.len != rows * t or observed.len != rows * t) return error.BadShape;
        if (h_patches < 1 or h_patches > cfg.block_size) return error.BadHorizon;

        // 1. causal per-patch normalization, broadcast to every step of the patch
        const loc = try alloc.alloc(f32, rows * t);
        const scale = try alloc.alloc(f32, rows * t);
        try causalPatchLocScale(inputs, observed, rows, t, ps, cfg.min_scale, loc, scale);

        // 2. asinh squash in each patch's frame
        const scaled = try alloc.alloc(f32, rows * t);
        for (0..rows * t) |i| {
            if (observed[i] == 0) {
                scaled[i] = 0;
            } else {
                const sc = @max(scale[i], std.math.floatEps(f32));
                const v = (inputs[i] - loc[i]) / sc;
                scaled[i] = if (cfg.use_asinh) asinh(v) else v;
            }
        }
        emit(hook, .scaled, 0, scaled);

        // 3. patch embedding over [values, missing], then the forecast-query tokens
        const x = try alloc.alloc(f32, rows * s * d);
        @memset(x, 0);
        const vec = try alloc.alloc(f32, 2 * ps);
        const hid = try alloc.alloc(f32, self.patch.hidden.rows);
        const dst = try alloc.alloc(f32, d);
        for (0..rows) |r| {
            for (0..patches) |p| {
                for (0..ps) |i| {
                    const step = r * t + p * ps + i;
                    vec[i] = scaled[step];
                    vec[ps + i] = if (observed[step] != 0) 0.0 else 1.0;
                }
                self.patch.forward(vec, hid, dst);
                @memcpy(x[(r * s + p) * d ..][0..d], dst);
            }
            for (0..h_patches) |p| {
                @memcpy(x[(r * s + patches + p) * d ..][0..d], self.forecast_query[0..d]);
            }
        }
        emit(hook, .x_in, 0, x);

        // 4. key validity, first observed patch, RoPE tables
        const key_valid = try alloc.alloc(u8, rows * patches);
        const first_obs = try alloc.alloc(usize, rows);
        try patchKeyValidity(observed, rows, patches, ps, cfg.attend_missing, key_valid, first_obs);
        const cos = try alloc.alloc(f32, rows * s * hd);
        const sin = try alloc.alloc(f32, rows * s * hd);
        try ropeTables(cfg, first_obs, rows, s, cos, sin);

        // 5. the transformer stack
        const ws = try Workspace.init(alloc, cfg, s);
        try self.transformerPass(x, rows, s, patches, key_valid, cos, sin, ws, hook);

        // 6. final norm, then the quantile head on the decoder tokens
        const npq = cfg.quantilesPerPatch();
        const hn = try alloc.alloc(f32, rows * s * d);
        for (0..rows) |r| {
            for (0..s) |tok| rmsNorm(x[(r * s + tok) * d ..][0..d], self.out_norm, hn[(r * s + tok) * d ..][0..d], cfg.norm_eps);
        }
        emit(hook, .out_norm, 0, hn);
        try self.headPredict(alloc, hn, rows, s, patches, h_patches, out, hook);

        // 7. back to raw space in the last context patch's frame
        for (0..rows) |r| {
            const lo = loc[r * t + t - 1];
            const sc = scale[r * t + t - 1];
            anchor_loc[r] = lo;
            anchor_scale[r] = sc;
            const span = h_patches * npq;
            for (0..h_patches * ps) |i| {
                const vals = out[r * span + i * cfg.n_knots ..][0..cfg.n_knots];
                for (vals) |*qv| {
                    var v = qv.*;
                    if (std.math.isNan(v)) continue;
                    if (cfg.use_asinh) v = sinh(v);
                    qv.* = v * sc + lo;
                }
                clampStep(vals);
            }
        }
        emit(hook, .block_quantiles, 0, out);
    }

    /// Run the layer stack in place over `(rows, s, d)` tokens.
    pub fn transformerPass(
        self: *const Model,
        x: []f32,
        rows: usize,
        s: usize,
        patches: usize,
        key_valid: []const u8,
        cos: []const f32,
        sin: []const f32,
        ws: Workspace,
        hook: ?Hook,
    ) !void {
        for (self.layers, 0..) |*layer, li| {
            for (layer.kind.axes(), 0..) |is_time, ai| {
                if (is_time) {
                    try timeAttention(self, layer, ai, x, rows, s, patches, key_valid, cos, sin, ws);
                } else {
                    varyAttention(self, layer, ai, x, rows, s, ws);
                }
            }
            mlpStep(self, layer, x, rows, s, ws);
            emit(hook, .layer, li, x);
        }
    }

    /// Quantile head over the normalized decoder tokens: raw head output, monotone
    /// quantiles, written to `out` `(rows, h_patches, patch_size, n_knots)`.
    fn headPredict(
        self: *const Model,
        alloc: std.mem.Allocator,
        hn: []const f32,
        rows: usize,
        s: usize,
        patches: usize,
        h_patches: usize,
        out: []f32,
        hook: ?Hook,
    ) !void {
        const cfg = self.cfg;
        const d = cfg.d_model;
        const npq = cfg.quantilesPerPatch();
        const raw = try alloc.alloc(f32, npq);
        const head_hid = try alloc.alloc(f32, self.head.hidden.rows);
        for (0..rows) |r| {
            for (0..h_patches) |p| {
                self.head.forward(hn[(r * s + patches + p) * d ..][0..d], head_hid, raw);
                for (0..cfg.patch_size) |i| monotoneQuantiles(cfg, raw[i * cfg.n_knots ..][0..cfg.n_knots]);
                @memcpy(out[(r * h_patches + p) * npq ..][0..npq], raw);
            }
        }
        emit(hook, .head_raw, 0, out);
    }
};

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------

pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// `log(1 + exp(x))` with torch's linear tail past 20.
pub fn softplus(x: f32) f32 {
    return if (x > 20.0) x else std.math.log1p(@exp(x));
}

pub fn asinh(x: f32) f32 {
    return std.math.asinh(x);
}

pub fn sinh(x: f32) f32 {
    return std.math.sinh(x);
}

/// Map a sinh overflow to the nearest finite value on the quantile axis, as the
/// reference's `clamp_nonfinite` does; a wholly non-finite step becomes zero.
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

pub fn rmsNorm(x: []const f32, scale: []const f32, out: []f32, eps: f32) void {
    var acc: f64 = 0;
    for (x) |v| acc += @as(f64, v) * @as(f64, v);
    const mean: f32 = @floatCast(acc / @as(f64, @floatFromInt(x.len)));
    const inv: f32 = 1.0 / @sqrt(mean + eps);
    for (x, scale, out) |v, s, *o| o.* = v * inv * s;
}

/// Non-crossing quantiles from one patch's raw head output, in place.
pub fn monotoneQuantiles(cfg: Config, raw: []f32) void {
    const q = cfg.n_knots;
    const m = cfg.medianIndex();
    const median = raw[m];
    var inc: [MAX_KNOTS]f32 = undefined;
    for (0..q) |i| {
        const w = if (i > m) cfg.knots[i] - cfg.knots[i - 1] else if (i < m) cfg.knots[i + 1] - cfg.knots[i] else 0.0;
        inc[i] = cfg.width_scale * cfg.width_factor * w * softplus(raw[i]);
    }
    var acc: f32 = 0;
    var i = m + 1;
    while (i < q) : (i += 1) {
        acc += inc[i];
        raw[i] = median + acc;
    }
    acc = 0;
    i = m;
    while (i > 0) {
        i -= 1;
        acc += inc[i];
        raw[i] = median - acc;
    }
    raw[m] = median;
}

/// Per-patch causal (loc, scale) broadcast to every step of the patch: the statistics at
/// each patch's last step, in f64 with Bessel's correction like the reference.
pub fn causalPatchLocScale(
    data: []const f32,
    observed: []const u8,
    rows: usize,
    t: usize,
    ps: usize,
    min_scale: f32,
    loc: []f32,
    scale: []f32,
) !void {
    if (t % ps != 0) return error.BadLength;
    for (0..rows) |r| {
        const xr = data[r * t ..][0..t];
        const mr = observed[r * t ..][0..t];
        var cum_w: f64 = 0;
        var cum_x: f64 = 0;
        var prev_mean: f64 = 0;
        var m2: f64 = 0;
        for (0..t) |i| {
            const w: f64 = if (mr[i] != 0) 1.0 else 0.0;
            const x: f64 = xr[i];
            cum_w += w;
            cum_x += w * x;
            const den = @max(cum_w, 1.0);
            const mean = cum_x / den;
            m2 += (x - prev_mean) * (x - mean) * w;
            prev_mean = mean;
            if ((i + 1) % ps == 0) {
                const p_loc: f32 = @floatCast(mean);
                const variance = m2 / @max(den - 1.0, 1.0);
                const p_scale: f32 = @floatCast(@max(@sqrt(@max(variance, 0.0)), @as(f64, min_scale)));
                const base = r * t + (i + 1 - ps);
                for (0..ps) |k| {
                    loc[base + k] = p_loc;
                    scale[base + k] = p_scale;
                }
            }
        }
    }
}

/// Per-patch key validity and each row's first patch holding an observation.
pub fn patchKeyValidity(
    observed: []const u8,
    rows: usize,
    patches: usize,
    ps: usize,
    attend_missing: bool,
    key_valid: []u8,
    first_obs: []usize,
) !void {
    if (observed.len != rows * patches * ps) return error.BadShape;
    for (0..rows) |r| {
        var started = false;
        var first: usize = 0;
        var seen = false;
        for (0..patches) |p| {
            var any = false;
            for (0..ps) |i| {
                if (observed[r * patches * ps + p * ps + i] != 0) {
                    any = true;
                    break;
                }
            }
            if (any and !seen) {
                first = p;
                seen = true;
            }
            started = started or any;
            key_valid[r * patches + p] = if (any or (attend_missing and started)) 1 else 0;
        }
        first_obs[r] = first;
    }
}

fn ropeTables(cfg: Config, first_obs: []const usize, rows: usize, s: usize, cos: []f32, sin: []f32) !void {
    const hd = cfg.headDim();
    if (hd > MAX_HEAD_DIM) return error.BadHeadDim;
    const half = hd / 2;
    var inv: [MAX_HEAD_DIM / 2]f32 = undefined;
    for (0..half) |i| {
        inv[i] = if (i < cfg.rope_dim / 2)
            @floatCast(1.0 / std.math.pow(f64, cfg.rope_theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(cfg.rope_dim))))
        else
            0.0;
    }
    for (0..rows) |r| {
        for (0..s) |tok| {
            const rel = @as(f32, @floatFromInt(tok)) - @as(f32, @floatFromInt(first_obs[r]));
            const base = (r * s + tok) * hd;
            for (0..half) |i| {
                const a = inv[i] * rel;
                const c = @cos(a);
                const sn = @sin(a);
                cos[base + i] = c;
                sin[base + i] = sn;
                cos[base + i + half] = c;
                sin[base + i + half] = sn;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// workspace and the transformer steps
// ---------------------------------------------------------------------------

const Workspace = struct {
    xn: []f32, // s*d  normalized token
    qa: []f32, // s*d  query per token, head-major
    ka: []f32, // s*d
    va: []f32, // s*d
    raw: []f32, // 3*d  fused projection of one token
    att: []f32, // s*d  attention output
    proj: []f32, // d
    gsc: []f32, // d
    scores: []f32, // s
    act2: []f32, // 2*mlp_hidden
    h1: []f32, // mlp_hidden
    hn: []f32, // s*d  out_norm
    head_hid: []f32, // 4*d

    fn init(alloc: std.mem.Allocator, cfg: Config, s: usize) !Workspace {
        const d = cfg.d_model;
        return .{
            .xn = try alloc.alloc(f32, s * d),
            .qa = try alloc.alloc(f32, s * d),
            .ka = try alloc.alloc(f32, s * d),
            .va = try alloc.alloc(f32, s * d),
            .raw = try alloc.alloc(f32, 3 * d),
            .att = try alloc.alloc(f32, s * d),
            .proj = try alloc.alloc(f32, d),
            .gsc = try alloc.alloc(f32, d),
            .scores = try alloc.alloc(f32, s),
            .act2 = try alloc.alloc(f32, 2 * cfg.mlp_hidden),
            .h1 = try alloc.alloc(f32, cfg.mlp_hidden),
            .hn = try alloc.alloc(f32, s * d),
            .head_hid = try alloc.alloc(f32, 4 * d),
        };
    }
};

fn applyRope(q: []f32, k: []f32, cos: []const f32, sin: []const f32, heads: usize, hd: usize) void {
    const half = hd / 2;
    for (0..heads) |h| {
        const base = h * hd;
        for (0..half) |i| {
            const q1 = q[base + i];
            const q2 = q[base + i + half];
            const k1 = k[base + i];
            const k2 = k[base + i + half];
            const c1 = cos[i];
            const s1 = sin[i];
            q[base + i] = q1 * c1 - q2 * s1;
            q[base + i + half] = q2 * c1 + q1 * s1;
            k[base + i] = k1 * c1 - k2 * s1;
            k[base + i + half] = k2 * c1 + k1 * s1;
        }
    }
}

fn keyAllowed(key_valid: []const u8, r: usize, patches: usize, is_dec: bool, tok: usize, kt: usize) bool {
    if (kt < patches) {
        if (is_dec) return key_valid[r * patches + kt] != 0;
        return (kt <= tok and key_valid[r * patches + kt] != 0) or kt == tok;
    }
    return is_dec and kt <= tok;
}

fn dotF32(a: []const f32, b: []const f32) f32 {
    var acc: f32 = 0;
    for (a, b) |x, y| acc += x * y;
    return acc;
}

fn timeAttention(
    m: *const Model,
    layer: *const Layer,
    ai: usize,
    x: []f32,
    rows: usize,
    s: usize,
    patches: usize,
    key_valid: []const u8,
    cos: []const f32,
    sin: []const f32,
    ws: Workspace,
) !void {
    const cfg = m.cfg;
    const d = cfg.d_model;
    const heads = cfg.num_heads;
    const hd = cfg.headDim();
    const attn = &layer.attn[ai];
    const norm_scale = layer.attn_norm[ai];
    const inv_sqrt_hd: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    for (0..rows) |r| {
        const xr = x[r * s * d ..][0 .. s * d];
        for (0..s) |tok| {
            rmsNorm(xr[tok * d ..][0..d], norm_scale, ws.xn[tok * d ..][0..d], cfg.norm_eps);
            attn.wqkv.gemv(ws.xn[tok * d ..][0..d], ws.raw[0 .. 3 * d]);
            for (0..hd) |i| {
                for (0..heads) |h| {
                    ws.qa[tok * d + h * hd + i] = ws.raw[i * heads + h];
                    ws.ka[tok * d + h * hd + i] = ws.raw[d + i * heads + h];
                    ws.va[tok * d + h * hd + i] = ws.raw[2 * d + i * heads + h];
                }
            }
            const cbase = (r * s + tok) * hd;
            applyRope(ws.qa[tok * d ..][0..d], ws.ka[tok * d ..][0..d], cos[cbase..][0..hd], sin[cbase..][0..hd], heads, hd);
        }

        for (0..s) |tok| {
            const is_dec = tok >= patches;
            for (0..heads) |head| {
                const qh = ws.qa[tok * d + head * hd ..][0..hd];
                var n: usize = 0;
                for (0..s) |kt| {
                    if (!keyAllowed(key_valid, r, patches, is_dec, tok, kt)) continue;
                    const kh = ws.ka[kt * d + head * hd ..][0..hd];
                    ws.scores[n] = dotF32(qh, kh) * inv_sqrt_hd;
                    n += 1;
                }
                var mx = ws.scores[0];
                for (ws.scores[1..n]) |sv| mx = @max(mx, sv);
                var sum: f32 = 0;
                for (ws.scores[0..n]) |*sv| {
                    sv.* = @exp(sv.* - mx);
                    sum += sv.*;
                }
                const inv_sum = 1.0 / sum;
                for (0..hd) |i| ws.att[tok * d + head * hd + i] = 0;
                var j: usize = 0;
                for (0..s) |kt| {
                    if (!keyAllowed(key_valid, r, patches, is_dec, tok, kt)) continue;
                    const w = ws.scores[j] * inv_sum;
                    j += 1;
                    const vh = ws.va[kt * d + head * hd ..][0..hd];
                    for (0..hd) |i| ws.att[tok * d + head * hd + i] += w * vh[i];
                }
            }
            if (attn.hasGate()) {
                attn.gate.gemv(ws.xn[tok * d ..][0..d], ws.gsc);
                for (0..d) |i| ws.att[tok * d + i] *= 1.0 / (1.0 + @exp(-ws.gsc[i]));
            }
            attn.wO.gemv(ws.att[tok * d ..][0..d], ws.proj);
            for (0..d) |i| xr[tok * d + i] += ws.proj[i];
        }
    }
}

fn varyAttention(m: *const Model, layer: *const Layer, ai: usize, x: []f32, rows: usize, s: usize, ws: Workspace) void {
    // Every row is its own group, so variate attention has exactly one key: itself.
    const cfg = m.cfg;
    const d = cfg.d_model;
    const heads = cfg.num_heads;
    const hd = cfg.headDim();
    const attn = &layer.attn[ai];
    const norm_scale = layer.attn_norm[ai];
    const vmat = attn.wqkv.rowView(2 * d, d);
    for (0..rows) |r| {
        const xr = x[r * s * d ..][0 .. s * d];
        for (0..s) |tok| {
            rmsNorm(xr[tok * d ..][0..d], norm_scale, ws.xn[tok * d ..][0..d], cfg.norm_eps);
            vmat.gemv(ws.xn[tok * d ..][0..d], ws.proj);
            for (0..hd) |i| {
                for (0..heads) |h| ws.att[tok * d + h * hd + i] = ws.proj[i * heads + h];
            }
            if (attn.hasGate()) {
                attn.gate.gemv(ws.xn[tok * d ..][0..d], ws.gsc);
                for (0..d) |i| ws.att[tok * d + i] *= 1.0 / (1.0 + @exp(-ws.gsc[i]));
            }
            attn.wO.gemv(ws.att[tok * d ..][0..d], ws.proj);
            for (0..d) |i| xr[tok * d + i] += ws.proj[i];
        }
    }
}

fn mlpStep(m: *const Model, layer: *const Layer, x: []f32, rows: usize, s: usize, ws: Workspace) void {
    const cfg = m.cfg;
    const d = cfg.d_model;
    for (0..rows) |r| {
        const xr = x[r * s * d ..][0 .. s * d];
        for (0..s) |tok| {
            rmsNorm(xr[tok * d ..][0..d], layer.mlp_norm, ws.xn[tok * d ..][0..d], cfg.norm_eps);
            layer.w12.gemv(ws.xn[tok * d ..][0..d], ws.act2[0 .. 2 * cfg.mlp_hidden]);
            for (0..cfg.mlp_hidden) |i| ws.h1[i] = ws.act2[i] * silu(ws.act2[cfg.mlp_hidden + i]);
            layer.w3.gemv(ws.h1[0..cfg.mlp_hidden], ws.proj);
            for (0..d) |i| xr[tok * d + i] += ws.proj[i];
        }
    }
}

fn residualParams(r: *const Residual) usize {
    return r.hidden.params() + r.hidden_b.len + r.out.params() + r.out_b.len + r.res.params() + r.res_b.len;
}

fn residualTernary(r: *const Residual) usize {
    var n: usize = 0;
    if (r.hidden.tern) n += r.hidden.params();
    if (r.out.tern) n += r.out.params();
    if (r.res.tern) n += r.res.params();
    return n;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "silu and softplus match their definitions" {
    try testing.expectApproxEqAbs(@as(f32, 0.0), silu(0.0), 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 0.6931472), softplus(0.0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 20.5), softplus(20.5), 1e-6);
}

test "rmsNorm normalizes and scales" {
    const x = [_]f32{ 1, -2, 3, 0 };
    const scale = [_]f32{ 1, 1, 2, 0.5 };
    var out: [4]f32 = undefined;
    rmsNorm(&x, &scale, &out, 1e-6);
    // rms = sqrt(14/4)
    const rms = @sqrt(@as(f32, 14.0 / 4.0));
    try testing.expectApproxEqAbs(1.0 / rms, out[0], 1e-5);
    try testing.expectApproxEqAbs(-2.0 / rms, out[1], 1e-5);
    try testing.expectApproxEqAbs(2.0 * 3.0 / rms, out[2], 1e-5);
}

test "monotone quantiles keep the median and fan out without crossing" {
    const cfg = config5m();
    var raw: [MAX_KNOTS]f32 = undefined;
    for (&raw, 0..) |*v, i| v.* = 0.01 * @as(f32, @floatFromInt(i)) - 0.1;
    const median = raw[cfg.medianIndex()];
    monotoneQuantiles(cfg, raw[0..cfg.n_knots]);
    try testing.expectEqual(median, raw[cfg.medianIndex()]);
    for (0..cfg.n_knots - 1) |i| try testing.expect(raw[i] <= raw[i + 1] + 1e-6);
}

test "ternary gemv is add/skip/subtract scaled per row" {
    const w = Matrix{
        .rows = 2,
        .cols = 3,
        .tern = true,
        .trits = @constCast(&[_]i8{ 1, 0, -1, -1, 1, 0 }),
        .scales = @constCast(&[_]f32{ 0.5, 2.0 }),
    };
    const x = [_]f32{ 1.0, 2.0, 3.0 };
    var y: [2]f32 = undefined;
    w.gemv(&x, &y);
    try testing.expectApproxEqAbs(@as(f32, -1.0), y[0], 1e-6); // (1 - 3) * 0.5
    try testing.expectApproxEqAbs(@as(f32, 2.0), y[1], 1e-6); // (-1 + 2) * 2
}
