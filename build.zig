const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize: std.builtin.OptimizeMode = if (b.release_mode != .off) .ReleaseFast else .Debug;

    const mod = b.createModule(.{
        .root_source_file = b.path("src/microgpt.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const exe = b.addExecutable(.{
        .name = "microgpt",
        .root_module = mod,
    });
    if (target.result.os.tag != .windows) {
        mod.linkSystemLibrary("m", .{});
    }
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| run_cmd.addArgs(args) else run_cmd.addArg("data/names.txt");
    const run_step = b.step("run", "Run microgpt");
    run_step.dependOn(&run_cmd.step);

    const c_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    c_mod.addCSourceFile(.{ .file = b.path("src/microgpt.c"), .flags = &[_][]const u8{
        "-Ofast",
        "-march=native",
        "-mavx2",
        "-mfma",
        "-Wall",
    } });
    const c_exe = b.addExecutable(.{
        .name = "microgpt-c",
        .root_module = c_mod,
    });
    if (target.result.os.tag != .windows) c_mod.linkSystemLibrary("m", .{});
    b.installArtifact(c_exe);

    const bitnet_mod = b.createModule(.{
        .root_source_file = b.path("src/bitnet.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const bitnet_exe = b.addExecutable(.{
        .name = "bitnet",
        .root_module = bitnet_mod,
    });
    if (target.result.os.tag != .windows) bitnet_mod.linkSystemLibrary("m", .{});
    b.installArtifact(bitnet_exe);

    const bitnet_cmd = b.addRunArtifact(bitnet_exe);
    if (b.args) |args| bitnet_cmd.addArgs(args) else bitnet_cmd.addArg("data/names.txt");
    const bitnet_step = b.step("bitnet", "Full pipeline: fp32 pretrain -> BitNet ternary QAT -> packed export");
    bitnet_step.dependOn(&bitnet_cmd.step);

    const ternary_test_mod = b.createModule(.{
        .root_source_file = b.path("src/ternary.zig"),
        .target = target,
        .optimize = optimize,
    });
    const ternary_tests = b.addTest(.{ .root_module = ternary_test_mod });
    const run_tests = b.addRunArtifact(ternary_tests);
    const test_step = b.step("test", "Run ternary packing unit tests");
    test_step.dependOn(&run_tests.step);
}
