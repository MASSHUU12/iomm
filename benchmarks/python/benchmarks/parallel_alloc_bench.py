from multiprocessing import Process, cpu_count


def parallel_alloc_benchmark():
    TotalAllocs = 10_000_000_000

    workers = cpu_count() or 1
    per_worker = TotalAllocs // workers

    class Small:
        __slots__ = ('x','y','z')
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    def worker():
        for _ in range(per_worker):
            _ = Small(1, 2, 3)

    processes = []
    for _ in range(workers):
        p = Process(target=worker)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def test_parallel_alloc_benchmark(benchmark) -> None:
    benchmark(parallel_alloc_benchmark)


if __name__ == "__main__":
    parallel_alloc_benchmark()
