const std = @import("std");

test "BenchmarkSmallReuse" {
    const ITERS = 10_000_000;
    const BYTES = 256;
    var buf: [BYTES]u8 = undefined;

    for (0..ITERS) |_| {
        for (0..BYTES) |i| {
            buf[i] = 0;
        }
    }
}
