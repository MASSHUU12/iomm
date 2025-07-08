import threading
import os

def parallel_alloc_benchmark():
    TotalAllocs = 1_000_000_000

    workers = os.cpu_count() or 1
    per_worker = TotalAllocs // workers

    class Small:
        __slots__ = ('x','y','z')
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    def worker():
        for _ in range(per_worker):
            _ = Small(1, 2, 3)

    threads = []
    for _ in range(workers):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def test_parallel_alloc_benchmark(benchmark) -> None:
    benchmark(parallel_alloc_benchmark)
