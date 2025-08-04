const std = @import("std");
const cpu = @import("get_cpu_count.zig");

const Queue = struct {
    items: std.ArrayList(i32),
    mutex: std.Thread.Mutex,
    cond: std.Thread.Condition,

    pub fn init(allocator: std.mem.Allocator) Queue {
        return Queue{
            .items = std.ArrayList(i32).init(allocator),
            .mutex = std.Thread.Mutex{},
            .cond = std.Thread.Condition{},
        };
    }

    pub fn deinit(self: *Queue) void {
        self.items.deinit();
    }

    pub fn push(self: *Queue, value: i32) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.items.append(value) catch return;
        self.cond.signal();
    }

    pub fn pop(self: *Queue) i32 {
        self.mutex.lock();
        defer self.mutex.unlock();

        while (self.items.items.len == 0) {
            self.cond.wait(&self.mutex);
        }

        const value = self.items.items[0];
        _ = self.items.orderedRemove(0);
        return value;
    }
};

fn producerFn(queue: *Queue, base: i32, count: usize) void {
    for (0..count) |i| {
        queue.push(base + @as(i32, @intCast(i)));
    }
}

fn consumerFn(queue: *Queue, count: usize) void {
    for (0..count) |_| {
        _ = queue.pop();
    }
}

test "BenchmarkSharedQueue" {
    const TotalOps = 1_000_000;

    const workers = cpu.getCpuCount();
    const producers = workers / 2;
    const consumers = workers - producers;

    const pushesPerProducer = TotalOps / 2 / producers;
    const popsPerConsumer = TotalOps / 2 / consumers;

    const allocator = std.heap.page_allocator;
    var q = Queue.init(allocator);
    defer q.deinit();

    var threads = std.ArrayList(std.Thread).init(allocator);
    defer threads.deinit();

    for (0..producers) |p| {
        const base = @as(i32, @intCast(p * pushesPerProducer));
        const thread = try std.Thread.spawn(.{}, producerFn, .{ &q, base, pushesPerProducer });
        try threads.append(thread);
    }

    for (0..consumers) |_| {
        const thread = try std.Thread.spawn(.{}, consumerFn, .{ &q, popsPerConsumer });
        try threads.append(thread);
    }

    for (threads.items) |thread| {
        thread.join();
    }
}
