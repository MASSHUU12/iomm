import numpy as np


def reuse_alloc_benchmark() -> None:
    ITERS = 10_000_000
    BYTES = 256

    buf = np.zeros(BYTES, dtype=np.uint8)
    for _ in range(ITERS):
        buf.fill(0)


def test_reuse_alloc(benchmark) -> None:
    benchmark(reuse_alloc_benchmark)


if __name__ == "__main__":
    reuse_alloc_benchmark()
