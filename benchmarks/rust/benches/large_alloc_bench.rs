use criterion::{criterion_group, criterion_main, Criterion};

const N: usize = 100_000;
const K: usize = 5 * 1_048_576;

fn bench_large_alloc(c: &mut Criterion) {
    c.bench_function("large_allocs", |b| {
        b.iter(|| {
            for _ in 0..N {
                let _v = vec![0u8; K];
            }
        });
    });
}

criterion_group!(large_alloc, bench_large_alloc);
criterion_main!(large_alloc);
