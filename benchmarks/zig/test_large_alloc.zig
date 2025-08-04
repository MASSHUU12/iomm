const std = @import("std");

test "BenchmarkLargeAllocDealloc" {
    const allocator = std.heap.page_allocator;
    const ITERS = 100_000;
    const BYTES = 5 * 1_048_576;

    for (0..ITERS) |_| {
        const a = allocator.alloc(u8, BYTES) catch return;
        defer allocator.free(a);
    }
}
