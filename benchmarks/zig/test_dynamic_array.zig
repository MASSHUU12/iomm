const std = @import("std");

test "BenchmarkDynamicArray" {
    const CAPACITY = 1_000_000;

    var sum: usize = 0;
    var arr = std.ArrayList(usize).init(std.heap.page_allocator);

    for (0..CAPACITY) |j| {
        arr.append(j) catch return;
    }

    for (arr.items) |v| {
        sum += v;
    }

    arr.deinit();
}
