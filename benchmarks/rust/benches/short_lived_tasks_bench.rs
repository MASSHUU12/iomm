use criterion::{criterion_group, criterion_main, Criterion};

fn bench_short_lived_tasks(c: &mut Criterion) {
    const M_TASKS: usize = 1_000_000_000;

    struct TaskData {
        a: i32,
        b: i32,
        c: i32,
    }

    c.bench_function("short_lived_tasks", |b| {
        b.iter(|| {
            let mut total = 0;
            for j in 0..M_TASKS {
                let j = std::hint::black_box(j);
                let t = std::hint::black_box(TaskData {
                    a: j as i32,
                    b: j as i32 * 2,
                    c: j as i32 * 3,
                });
                total += t.a + t.b + t.c;
            }
            std::hint::black_box(total);
        });
    });
}

criterion_group!(short_lived_task, bench_short_lived_tasks);
criterion_main!(short_lived_task);
