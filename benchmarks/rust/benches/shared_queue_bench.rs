use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::thread;
use std::thread::available_parallelism;

struct Queue {
    items: Mutex<Vec<i32>>,
    cond: Condvar,
}

impl Queue {
    fn new() -> Self {
        Queue {
            items: Mutex::new(Vec::with_capacity(1024)),
            cond: Condvar::new(),
        }
    }

    fn push(&self, v: i32) {
        let mut items = self.items.lock().unwrap();
        items.push(v);
        self.cond.notify_one();
    }

    fn pop(&self) -> i32 {
        let mut items = self.items.lock().unwrap();
        while items.is_empty() {
            items = self.cond.wait(items).unwrap();
        }
        let v = items[0];
        items.remove(0);
        v
    }
}

fn bench_shared_queue(c: &mut Criterion) {
    const TOTAL_OPS: usize = 1_000_000;

    let num_cpus = available_parallelism().unwrap().get();
    let producers = num_cpus / 2;
    let consumers = num_cpus - producers;

    let pushes_per_producer = TOTAL_OPS / 2 / producers;
    let pops_per_consumer = TOTAL_OPS / 2 / consumers;

    c.bench_function("shared_queue", |b| {
        b.iter(|| {
            let q = Arc::new(Queue::new());
            let barrier = Arc::new(Barrier::new(producers + consumers + 1));
            let mut handles = Vec::with_capacity(producers + consumers);

            for p in 0..producers {
                let q_clone = Arc::clone(&q);
                let barrier_clone = Arc::clone(&barrier);
                handles.push(thread::spawn(move || {
                    barrier_clone.wait();
                    let base = p * pushes_per_producer;
                    for i in 0..pushes_per_producer {
                        q_clone.push(black_box((base + i) as i32));
                    }
                    0i32
                }));
            }

            for _ in 0..consumers {
                let q_clone = Arc::clone(&q);
                let barrier_clone = Arc::clone(&barrier);
                handles.push(thread::spawn(move || {
                    barrier_clone.wait();
                    let mut sum = 0i32;
                    for _ in 0..pops_per_consumer {
                        sum += q_clone.pop();
                    }
                    black_box(sum)
                }));
            }

            barrier.wait();

            let mut grand_sum = 0i32;
            for (idx, handle) in handles.into_iter().enumerate() {
                let thread_sum = handle.join().unwrap();
                if idx >= producers {
                    grand_sum += thread_sum;
                }
            }
            black_box(grand_sum);
        });
    });
}

criterion_group!(shared_queue, bench_shared_queue);
criterion_main!(shared_queue);
