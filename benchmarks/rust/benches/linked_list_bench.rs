use criterion::{criterion_group, criterion_main, Criterion};
use std::ptr;

fn bench_linked_list(c: &mut Criterion) {
    const M: usize = 1_000_000;

    struct Node {
        value: i32,
        next: *mut Node,
    }

    c.bench_function("linked_list", |b| {
        b.iter(|| {
            let mut head: *mut Node = ptr::null_mut();

            for j in 0..M {
                let new_node = Box::into_raw(Box::new(Node {
                    value: j as i32,
                    next: ptr::null_mut(),
                }));

                unsafe {
                    (*new_node).next = head;
                    head = new_node;
                }
            }

            let mut sum = 0;
            unsafe {
                let mut current = head;
                while !current.is_null() {
                    sum += (*current).value;
                    current = (*current).next;
                }
            }

            unsafe {
                let mut current = head;
                while !current.is_null() {
                    let next = (*current).next;
                    let _ = Box::from_raw(current);
                    current = next;
                }
            }

            std::hint::black_box(sum);
        });
    });
}

criterion_group!(linked_list, bench_linked_list);
criterion_main!(linked_list);
