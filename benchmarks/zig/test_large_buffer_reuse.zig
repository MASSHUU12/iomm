const std = @import("std");

test "BenchmarkLargeReuse" {
    const ITERS = 100_000;
    const BYTES = 5 * 1_048_576;
    var buf: [BYTES]u8 = undefined;

    for (0..ITERS) |_| {
        for (0..BYTES) |i| {
            buf[i] = 0;
        }
    }
}
