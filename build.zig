const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize: std.builtin.OptimizeMode = if (b.release_mode != .off) .ReleaseFast else .Debug;

    // The framework: the shared train -> quantize -> deploy core
    // (RNG, ternary codec, absmean QAT, Adam, packed-file container, clock).
    // Every example imports it as "framework"; the test step runs its units.
    const fw_mod = b.createModule(.{
        .root_source_file = b.path("src/framework.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // fp32 reference baseline: the original microGPT in Zig, A/B twin of
    // c/microgpt.c and the sanity check for the framework's fp32 pretrain leg
    const fp32_mod = b.createModule(.{
        .root_source_file = b.path("src/fp32.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const fp32_exe = b.addExecutable(.{
        .name = "fp32-baseline",
        .root_module = fp32_mod,
    });
    if (target.result.os.tag != .windows) fp32_mod.linkSystemLibrary("m", .{});
    b.installArtifact(fp32_exe);

    const fp32_cmd = b.addRunArtifact(fp32_exe);
    if (b.args) |args| fp32_cmd.addArgs(args) else fp32_cmd.addArg("data/names.txt");
    const run_step = b.step("run", "Run the fp32 baseline trainer");
    run_step.dependOn(&fp32_cmd.step);

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

    // Examples: thin per-model drivers over the framework. Each one runs the
    // whole pipeline -- fp32 pretrain -> ternary QAT -> validation -> packed
    // export -> round-trip check -- and writes its artifacts to modeldata/.
    const gpt_mod = b.createModule(.{
        .root_source_file = b.path("examples/gpt_names.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    gpt_mod.addImport("framework", fw_mod);
    const gpt_exe = b.addExecutable(.{
        .name = "example-gpt",
        .root_module = gpt_mod,
    });
    if (target.result.os.tag != .windows) gpt_mod.linkSystemLibrary("m", .{});
    b.installArtifact(gpt_exe);

    const gpt_cmd = b.addRunArtifact(gpt_exe);
    if (b.args) |args| gpt_cmd.addArgs(args) else gpt_cmd.addArg("data/names.txt");
    const gpt_step = b.step("example-gpt", "Example: micro-GPT names (train -> quantize -> pack)");
    gpt_step.dependOn(&gpt_cmd.step);

    const cnn_mod = b.createModule(.{
        .root_source_file = b.path("examples/gesture_cnn.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    cnn_mod.addImport("framework", fw_mod);
    const cnn_exe = b.addExecutable(.{
        .name = "example-cnn",
        .root_module = cnn_mod,
    });
    if (target.result.os.tag != .windows) cnn_mod.linkSystemLibrary("m", .{});
    b.installArtifact(cnn_exe);

    const cnn_cmd = b.addRunArtifact(cnn_exe);
    const cnn_step = b.step("example-cnn", "Example: TinyML gesture 1D-CNN (train -> quantize -> pack)");
    cnn_step.dependOn(&cnn_cmd.step);

    const ncps_mod = b.createModule(.{
        .root_source_file = b.path("examples/liquid_sine.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    ncps_mod.addImport("framework", fw_mod);
    const ncps_exe = b.addExecutable(.{
        .name = "example-ncps",
        .root_module = ncps_mod,
    });
    if (target.result.os.tag != .windows) ncps_mod.linkSystemLibrary("m", .{});
    b.installArtifact(ncps_exe);

    const ncps_cmd = b.addRunArtifact(ncps_exe);
    const ncps_step = b.step("example-ncps", "Example: CfC/LTC Neural Circuit Policies (train -> quantize -> pack)");
    ncps_step.dependOn(&ncps_cmd.step);

    // Example 4: deploy Mistral's Timestral-5M time-series model behind the
    // sketch-a-series app, then quantize it with this framework and re-run the
    // same app against the ternary artifact.
    const ts_lib = b.createModule(.{
        .root_source_file = b.path("src/timestral.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    ts_lib.addImport("framework", fw_mod);
    if (target.result.os.tag != .windows) ts_lib.linkSystemLibrary("m", .{});
    const ts_mod = b.createModule(.{
        .root_source_file = b.path("examples/timestral_sketch.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    ts_mod.addImport("timestral", ts_lib);
    ts_mod.addImport("framework", fw_mod);
    const ts_exe = b.addExecutable(.{
        .name = "timestral-cli",
        .root_module = ts_mod,
    });
    if (target.result.os.tag != .windows) ts_mod.linkSystemLibrary("m", .{});
    b.installArtifact(ts_exe);

    const ts_quant_cmd = b.addRunArtifact(ts_exe);
    ts_quant_cmd.addArgs(&.{ "quantize", "modeldata/timestral-5m/model.safetensors", "modeldata/timestral_packed.bin" });
    const ts_quant_step = b.step("timestral-quantize", "Quantize Timestral-5M to the framework's ternary artifact");
    ts_quant_step.dependOn(&ts_quant_cmd.step);

    const ts_eval_cmd = b.addRunArtifact(ts_exe);
    ts_eval_cmd.addArgs(&.{ "eval", "modeldata/timestral_packed.bin", "--json", "modeldata/timestral_eval_ternary.json" });
    const ts_eval_step = b.step("timestral-eval", "Score the ternary Timestral artifact (pinball, MASE, latency)");
    ts_eval_step.dependOn(&ts_eval_cmd.step);

    const ts_fp32_cmd = b.addRunArtifact(ts_exe);
    ts_fp32_cmd.addArgs(&.{ "eval", "modeldata/timestral-5m/model.safetensors", "--json", "modeldata/timestral_eval_fp32.json" });
    const ts_fp32_step = b.step("timestral-eval-fp32", "Score the fp32 Timestral checkpoint with the same engine");
    ts_fp32_step.dependOn(&ts_fp32_cmd.step);

    const ts_serve_cmd = b.addRunArtifact(ts_exe);
    ts_serve_cmd.addArgs(&.{ "serve", "modeldata/timestral_packed.bin" });
    const ts_serve_step = b.step("timestral-serve", "Run the ternary forecast worker (JSON lines on stdin/stdout)");
    ts_serve_step.dependOn(&ts_serve_cmd.step);

    const ts_all_step = b.step("example-timestral", "Example: Timestral-5M sketch-a-series (quantize -> eval -> serve)");
    ts_all_step.dependOn(&ts_quant_cmd.step);
    ts_all_step.dependOn(&ts_eval_cmd.step);

    // unit tests: the framework's own modules plus each example's suite
    const fw_tests = b.addTest(.{ .root_module = fw_mod });
    const run_fw_tests = b.addRunArtifact(fw_tests);

    const gpt_test_mod = b.createModule(.{
        .root_source_file = b.path("examples/gpt_names.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    gpt_test_mod.addImport("framework", fw_mod);
    const gpt_tests = b.addTest(.{ .root_module = gpt_test_mod });
    const run_gpt_tests = b.addRunArtifact(gpt_tests);

    const cnn_test_mod = b.createModule(.{
        .root_source_file = b.path("examples/gesture_cnn.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    cnn_test_mod.addImport("framework", fw_mod);
    const cnn_tests = b.addTest(.{ .root_module = cnn_test_mod });
    const run_cnn_tests = b.addRunArtifact(cnn_tests);

    const ncps_test_mod = b.createModule(.{
        .root_source_file = b.path("examples/liquid_sine.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    ncps_test_mod.addImport("framework", fw_mod);
    const ncps_tests = b.addTest(.{ .root_module = ncps_test_mod });
    const run_ncps_tests = b.addRunArtifact(ncps_tests);

    const ts_test_mod = b.createModule(.{
        .root_source_file = b.path("examples/timestral_sketch.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    ts_test_mod.addImport("timestral", ts_lib);
    ts_test_mod.addImport("framework", fw_mod);
    const ts_tests = b.addTest(.{ .root_module = ts_test_mod });
    const run_ts_tests = b.addRunArtifact(ts_tests);

    const ts_lib_tests = b.addTest(.{ .root_module = ts_lib });
    const run_ts_lib_tests = b.addRunArtifact(ts_lib_tests);

    const test_step = b.step("test", "Run framework + example unit tests");
    test_step.dependOn(&run_fw_tests.step);
    test_step.dependOn(&run_gpt_tests.step);
    test_step.dependOn(&run_cnn_tests.step);
    test_step.dependOn(&run_ncps_tests.step);
    test_step.dependOn(&run_ts_tests.step);
    test_step.dependOn(&run_ts_lib_tests.step);
}
