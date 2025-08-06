import numpy as np


def large_mem_reuse_alloc_benchmark() -> None:
    ITERS = 100_000
    BYTES = 5 * 1_048_576

    buf = np.zeros(BYTES, dtype=np.uint8)
    for _ in range(ITERS):
        buf.fill(0)


def test_large_mem_reuse_alloc_benchmark(benchmark) -> None:
    benchmark(large_mem_reuse_alloc_benchmark)


if __name__ == "__main__":
    large_mem_reuse_alloc_benchmark()
