use criterion::{criterion_group, criterion_main, Criterion};

const N: usize = 10_000_000;
const K: usize = 256;

fn bench_large_buffer_reuse(c: &mut Criterion) {
    let mut buf = vec![0u8; K];
    c.bench_function("large_buffer_reuse", |b| {
        b.iter(|| {
            for _ in 0..N {
                for i in 0..buf.len() {
                    buf[i] = 0;
                }
            }
        })
    });
}

criterion_group!(large_buffer_reuse, bench_large_buffer_reuse);
criterion_main!(large_buffer_reuse);
