use criterion::{criterion_group, criterion_main, Criterion};

/// A silly recursive fib for demo purposes.
fn fibonacci(n: u64) -> u64 {
    match n {
        0 | 1 => n,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn bench_fibonacci(c: &mut Criterion) {
    let mut group = c.benchmark_group("fib");
    // measure fib(10), fib(20) and fib(30)
    for &i in &[10u64, 20, 30] {
        group.bench_with_input(format!("fib_{}", i), &i, |b, &i| {
            b.iter(|| fibonacci(i));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_fibonacci);
criterion_main!(benches);
