const std = @import("std");

const Node = struct {
    value: usize,
    next: ?*Node,
};

test "BenchmarkLinkedList" {
    const M = 100_000_000;

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const allocator = arena.allocator();

    var sum: usize = 0;
    var head: ?*Node = null;

    for (0..M) |j| {
        const n = try allocator.create(Node);
        n.* = Node{ .value = j, .next = head };
        head = n;
    }

    var cur = head;
    while (cur) |node| {
        sum += node.value;
        cur = node.next;
    }

    std.mem.doNotOptimizeAway(&sum);
}
