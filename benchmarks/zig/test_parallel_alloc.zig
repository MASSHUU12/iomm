const std = @import("std");
const cpu = @import("get_cpu_count.zig");

const Small = struct {
    x: i64,
    y: i64,
    z: i64,
};

test "BenchmarkParallelAlloc" {
    const TotalAllocs = 1_000_000_000;

    const allocator = std.heap.page_allocator;

    const workers = cpu.getCpuCount();
    const perWorker = TotalAllocs / workers;

    var threads = std.ArrayList(std.Thread).init(allocator);
    defer threads.deinit();

    for (0..workers) |_| {
        _ = threads.append(try std.Thread.spawn(.{}, workerFn, .{perWorker})) catch return;
    }

    for (threads.items) |thread| {
        thread.join();
    }
}

fn workerFn(perWorker: usize) void {
    for (0..perWorker) |i| {
        _ = &Small{
            .x = @intCast(i * 1),
            .y = @intCast(i * 2),
            .z = @intCast(i * 3),
        };
    }
}
