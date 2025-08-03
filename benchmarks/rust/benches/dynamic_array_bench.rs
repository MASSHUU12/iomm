use criterion::{criterion_group, criterion_main, Criterion};

fn bench_dynamic_array(c: &mut Criterion) {
    const CAPACITY: usize = 1_000_000;

    c.bench_function("dynamic_array", |b| {
        b.iter(|| {
            let mut arr = Vec::with_capacity(CAPACITY);

            for j in 0..CAPACITY {
                arr.push(j);
            }

            let mut sum = 0;
            for &v in &arr {
                sum += v;
            }

            std::hint::black_box(sum);
        });
    });
}

criterion_group!(dynamic_array, bench_dynamic_array);
criterion_main!(dynamic_array);
