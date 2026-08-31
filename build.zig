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
}
