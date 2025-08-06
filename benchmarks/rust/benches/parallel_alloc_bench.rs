use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::thread;
use std::thread::available_parallelism;

fn bench_parallel_alloc(c: &mut Criterion) {
    const TOTAL_ALLOCS: usize = 1_000_000_000;

    struct Small {
        x: i64,
        y: i64,
        z: i64,
    }

    let num_cpus = available_parallelism().unwrap().get();
    let per_worker = TOTAL_ALLOCS / num_cpus;

    c.bench_function("parallel_alloc", |b| {
        b.iter(|| {
            let barrier = Arc::new(Barrier::new(num_cpus + 1));
            let mut handles = Vec::with_capacity(num_cpus);

            for _ in 0..num_cpus {
                let barrier_clone = Arc::clone(&barrier);
                handles.push(thread::spawn(move || {
                    barrier_clone.wait();
                    let mut sum = 0i64;
                    for i in 0..per_worker {
                        let s = Small { x: 1, y: 2, z: 3 };
                        sum += s.x + s.y + s.z + (i as i64);
                    }
                    sum
                }));
            }

            barrier.wait();

            let mut grand_sum = 0i64;
            for handle in handles {
                let sum = handle.join().unwrap();
                grand_sum += sum;
            }
            black_box(grand_sum);
        });
    });
}

criterion_group!(parallel_alloc, bench_parallel_alloc);
criterion_main!(parallel_alloc);
