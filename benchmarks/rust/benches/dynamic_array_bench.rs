use criterion::{criterion_group, criterion_main, Criterion};

fn bench_dynamic_array(c: &mut Criterion) {
    const CAPACITY: usize = 100_000_000;

    c.bench_function("dynamic_array", |b| {
        b.iter(|| {
            let mut arr = Vec::new();

            for j in 0..CAPACITY {
                arr.push(j);
            }

            let sum: usize = arr.iter().copied().sum();
            std::hint::black_box(sum);
            std::hint::black_box(&arr);
        });
    });
}

criterion_group!(dynamic_array, bench_dynamic_array);
criterion_main!(dynamic_array);
