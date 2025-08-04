const std = @import("std");

test "BenchmarkSmallAllocDealloc" {
    const allocator = std.heap.page_allocator;
    const ITERS = 10_000_000;
    const BYTES = 256;

    for (0..ITERS) |_| {
        const a = allocator.alloc(u8, BYTES) catch return;
        defer allocator.free(a);
    }
}
