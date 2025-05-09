const std = @import("std");

fn fibonacci(n: u64) u64 {
    if (n < 2) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

fn bench_job() void {
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        _ = fibonacci(30);
    }
}

pub fn main() !void {
    bench_job();
}
