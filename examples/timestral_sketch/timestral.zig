//! Timestral-5M deployment module: the third-party model this repository deploys,
//! quantizes and serves.
//!
//!   model      the patch transformer forward pass (fp32 or ternary weights)
//!   weights    safetensors loading + the framework's absmean ternary quantizer
//!   artifact   the packed deployable artifact (config + fp32 tensors + pinned trits)
//!   engine     request-shaped decoding: padding, clampsym, refine frames, capping
//!   eval       deterministic stroke corpus and pinball/MASE metrics
//!   fixture    reader for the PyTorch reference dumps the parity test compares against
//!
//! The example driver lives in `examples/timestral_sketch.zig`; the web app in `web/`
//! talks to it through `serve`'s JSON-lines protocol.

const std = @import("std");

pub const model = @import("timestral/model.zig");
pub const weights = @import("timestral/weights.zig");
pub const artifact = @import("timestral/artifact.zig");
pub const engine = @import("timestral/engine.zig");
pub const eval = @import("timestral/eval.zig");
pub const fixture = @import("timestral/fixture.zig");

test {
    std.testing.refAllDecls(@This());
}
