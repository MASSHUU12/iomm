const std = @import("std");

test "BenchmarkShortLivedTasks" {
    const MTasks = 100_000_000;

    const TaskData = struct {
        a: usize,
        b: usize,
        c: usize,
    };

    var sum: usize = 0;
    for (0..MTasks) |j| {
        const t = TaskData{
            .a = j,
            .b = j * 2,
            .c = j * 3,
        };

        sum += t.a + t.b + t.c;
    }
}
