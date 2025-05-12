use criterion::{criterion_group, criterion_main, Criterion};

/// Number of iterations for small allocs
const N_SMALL: usize = 1_000_000;
/// Size of each small allocation (bytes)
const SMALL_K: usize = 64;

/// Benchmark: reuse a single buffer instead of reallocating
fn bench_buffer_reuse(c: &mut Criterion) {
    // Pre-allocate once
    let mut buf = vec![0u8; SMALL_K];
    c.bench_function("reused_buffer", |b| {
        b.iter(|| {
            for _ in 0..N_SMALL {
                // reuse the same buffer: clear and refill
                buf.clear();
                buf.resize(SMALL_K, 0);
            }
        })
    });
}

criterion_group!(micro_buffer_reuse, bench_buffer_reuse);
criterion_main!(micro_buffer_reuse);
