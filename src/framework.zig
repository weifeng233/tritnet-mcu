//! TritNet-MCU framework: the shared train -> quantize -> deploy core.
//!
//! Everything the examples have in common lives here, in pure Zig with
//! nothing beyond libc:
//!
//!   rng          deterministic xorshift stream (uniform / gauss / range)
//!   ternary      the 5-trits-per-byte weight codec (pack5 / unpack5)
//!   quant        BitNet absmean ternary QAT: quantTrits, STE grad mask,
//!                8-bit activation quantization, wired-only scale variant
//!   adam         SIMD Adam update + cosine LR schedule + bias correction
//!   packed_model little-endian model-file writers/reader and the shared
//!                ternary-matrix record used by every packed artifact
//!   timing       portable monotonic clock
//!
//! The pipeline every consumer follows: fp32 pretrain -> ternary QAT ->
//! validation -> packed export -> C deployment check. The examples in
//! examples/ are thin per-model drivers over these modules; the C engines
//! in c/ consume the packed artifacts this container defines.

const std = @import("std");

pub const rng = @import("framework/rng.zig");
pub const ternary = @import("framework/ternary.zig");
pub const quant = @import("framework/quant.zig");
pub const adam = @import("framework/adam.zig");
pub const packed_model = @import("framework/packed_model.zig");
pub const timing = @import("framework/timing.zig");

test {
    std.testing.refAllDecls(@This());
}
