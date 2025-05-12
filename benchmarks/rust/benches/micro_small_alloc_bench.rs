use criterion::{criterion_group, criterion_main, Criterion};

/// Number of iterations for small allocs
const N_SMALL: usize = 1_000_000;
/// Size of each small allocation (bytes)
const SMALL_K: usize = 64;

/// Benchmark: small allocations/deallocations
fn bench_small_alloc(c: &mut Criterion) {
    c.bench_function("fresh_small_allocs", |b| {
        b.iter(|| {
            for _ in 0..N_SMALL {
                // allocate a small Vec and drop
                let _v = vec![0u8; SMALL_K];
            }
        })
    });
}

criterion_group!(micro_small_alloc, bench_small_alloc);
criterion_main!(micro_small_alloc);
