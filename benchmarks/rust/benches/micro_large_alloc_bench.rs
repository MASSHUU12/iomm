use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

/// Number of iterations for large allocs
const N_LARGE: usize = 10_000;
/// Size of each large allocation (bytes)
const LARGE_M: usize = 100 * 1024 * 1024; // 100 MB

/// Benchmark: allocate and free a large block repeatedly
fn bench_large_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_allocs");
    for &size in &[LARGE_M] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}MB", size / (1024 * 1024))),
            &size,
            |b, &s| {
                b.iter(|| {
                    for _ in 0..N_LARGE {
                        let _v = vec![0u8; s];
                    }
                })
            },
        );
    }
    group.finish();
}

criterion_group!(micro_large_alloc, bench_large_alloc);
criterion_main!(micro_large_alloc);
