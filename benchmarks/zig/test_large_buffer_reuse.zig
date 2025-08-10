const std = @import("std");

test "BenchmarkLargeReuse" {
    const ITERS = 100_000;
    const BYTES = 5 * 1_048_576;
    var buf: [BYTES]u8 = undefined;

    for (0..ITERS) |iter| {
        for (0..BYTES) |i| {
            buf[i] = @truncate((iter + i) & 0xFF);
        }
        std.mem.doNotOptimizeAway(&buf);
    }
}
