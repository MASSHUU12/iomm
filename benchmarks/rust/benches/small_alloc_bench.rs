use criterion::{criterion_group, criterion_main, Criterion};

const N: usize = 10_000_000;
const K: usize = 256;

fn bench_small_alloc(c: &mut Criterion) {
    c.bench_function("small_allocs", |b| {
        b.iter(|| {
            for _ in 0..N {
                let _v = vec![0u8; K];
            }
        })
    });
}

criterion_group!(small_alloc, bench_small_alloc);
criterion_main!(small_alloc);
