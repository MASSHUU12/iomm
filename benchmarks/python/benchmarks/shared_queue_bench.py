import threading
import os

class SharedQueue:
    def __init__(self, initial_capacity=1024):
        self.items = []
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

    def push(self, v: int):
        with self.lock:
            self.items.append(v)
            self.cond.notify()

    def pop(self) -> int:
        with self.lock:
            while not self.items:
                self.cond.wait()
            v = self.items.pop(0)
            return v


def shared_queue_benchmark():
    TotalOps = 1_000_000
    workers = os.cpu_count() or 1
    producers = workers // 2 or 1
    consumers = workers - producers or 1

    pushes_per_producer = (TotalOps // 2) // producers
    pops_per_consumer = (TotalOps // 2) // consumers

    q = SharedQueue()

    def producer(seed: int):
        base = seed * pushes_per_producer
        for i in range(pushes_per_producer):
            q.push(base + i)

    def consumer():
        for _ in range(pops_per_consumer):
            _ = q.pop()

    threads = []

    for p in range(producers):
        t = threading.Thread(target=producer, args=(p,))
        t.start()
        threads.append(t)
    for _ in range(consumers):
        t = threading.Thread(target=consumer)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def test_shared_queue_benchmark(benchmark) -> None:
    benchmark(shared_queue_benchmark)
