const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize: std.builtin.OptimizeMode = if (b.release_mode != .off) .ReleaseFast else .Debug;

    // fp32 baseline trainer/inference (the original microGPT, in Zig)
    const fp32_mod = b.createModule(.{
        .root_source_file = b.path("src/fp32.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const fp32_exe = b.addExecutable(.{
        .name = "tritnet-fp32",
        .root_module = fp32_mod,
    });
    if (target.result.os.tag != .windows) {
        fp32_mod.linkSystemLibrary("m", .{});
    }
    b.installArtifact(fp32_exe);

    const run_cmd = b.addRunArtifact(fp32_exe);
    if (b.args) |args| run_cmd.addArgs(args) else run_cmd.addArg("data/names.txt");
    const run_step = b.step("run", "Run the fp32 baseline trainer");
    run_step.dependOn(&run_cmd.step);

    // C reference implementation, kept for A/B comparison
    const c_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    c_mod.addCSourceFile(.{ .file = b.path("c/microgpt.c"), .flags = &[_][]const u8{
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

    // full pipeline: fp32 pretrain -> ternary QAT -> ternary-packed export
    const tritnet_mod = b.createModule(.{
        .root_source_file = b.path("src/tritnet.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const tritnet_exe = b.addExecutable(.{
        .name = "tritnetmcu",
        .root_module = tritnet_mod,
    });
    if (target.result.os.tag != .windows) tritnet_mod.linkSystemLibrary("m", .{});
    b.installArtifact(tritnet_exe);

    const tritnet_cmd = b.addRunArtifact(tritnet_exe);
    if (b.args) |args| tritnet_cmd.addArgs(args) else tritnet_cmd.addArg("data/names.txt");
    const tritnet_step = b.step("tritnet", "Full pipeline: fp32 pretrain -> ternary QAT -> packed export");
    tritnet_step.dependOn(&tritnet_cmd.step);

    // TinyML 1D-CNN: fp32 pretrain -> ternary QAT -> packed export + verification
    const cnn_mod = b.createModule(.{
        .root_source_file = b.path("src/cnn1d.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const cnn_exe = b.addExecutable(.{
        .name = "tritnet-cnn",
        .root_module = cnn_mod,
    });
    if (target.result.os.tag != .windows) cnn_mod.linkSystemLibrary("m", .{});
    b.installArtifact(cnn_exe);

    const cnn_cmd = b.addRunArtifact(cnn_exe);
    const cnn_step = b.step("cnn", "TinyML 1D-CNN: fp32 pretrain -> ternary QAT -> packed export");
    cnn_step.dependOn(&cnn_cmd.step);

    const ternary_test_mod = b.createModule(.{
        .root_source_file = b.path("src/ternary.zig"),
        .target = target,
        .optimize = optimize,
    });
    const ternary_tests = b.addTest(.{ .root_module = ternary_test_mod });
    const run_tests = b.addRunArtifact(ternary_tests);

    const cnn_test_mod = b.createModule(.{
        .root_source_file = b.path("src/cnn1d.zig"),
        .target = target,
        .optimize = optimize,
    });
    const cnn_tests = b.addTest(.{ .root_module = cnn_test_mod });
    const run_cnn_tests = b.addRunArtifact(cnn_tests);

    const test_step = b.step("test", "Run ternary packing + 1D-CNN unit tests");
    test_step.dependOn(&run_tests.step);
    test_step.dependOn(&run_cnn_tests.step);
}
