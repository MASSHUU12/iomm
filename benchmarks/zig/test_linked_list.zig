const std = @import("std");

const Node = struct {
    value: usize,
    next: ?*Node,
};

test "BenchmarkLinkedList" {
    const M = 1_000_000;

    var sum: usize = 0;
    var head: ?*Node = null;

    for (0..M) |j| {
        const n = std.heap.page_allocator.create(Node) catch return;
        n.* = Node{ .value = j, .next = head };
        head = n;
    }

    var cur = head;
    while (cur) |node| {
        sum += node.value;
        cur = node.next;
    }

    cur = head;
    while (cur) |node| {
        const next = node.next;
        std.heap.page_allocator.destroy(node);
        cur = next;
    }
}
