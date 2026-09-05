//! Wall-clock timing shared by the trainer and the latency benchmarks.
//! QueryPerformanceCounter on Windows, CLOCK_MONOTONIC everywhere else.

const std = @import("std");
const builtin = @import("builtin");

const winclib = if (builtin.os.tag == .windows) @cImport({
    @cInclude("windows.h");
}) else struct {};
const timeclib = if (builtin.os.tag != .windows) @cImport({
    @cInclude("time.h");
}) else struct {};

/// Monotonic seconds, for elapsed-time math only (not an epoch).
pub fn nowS() f64 {
    if (builtin.os.tag == .windows) {
        var freq: winclib.LARGE_INTEGER = undefined;
        var cnt: winclib.LARGE_INTEGER = undefined;
        _ = winclib.QueryPerformanceFrequency(&freq);
        _ = winclib.QueryPerformanceCounter(&cnt);
        return @as(f64, @floatFromInt(cnt.QuadPart)) / @as(f64, @floatFromInt(freq.QuadPart));
    } else {
        var ts: timeclib.timespec = undefined;
        _ = timeclib.clock_gettime(timeclib.CLOCK_MONOTONIC, &ts);
        return @as(f64, @floatFromInt(ts.tv_sec)) + @as(f64, @floatFromInt(ts.tv_nsec)) * 1e-9;
    }
}

test "clock advances" {
    const a = nowS();
    var spins: usize = 0;
    while (spins < 1 << 26 and nowS() == a) : (spins += 1) {}
    try std.testing.expect(nowS() > a);
}
