def small_alloc_dealloc_benchmark() -> None:
    ITERS = 100_000
    BYTES = 256

    for _ in range(0, ITERS):
        _ = bytearray(BYTES)

def test_small_alloc_dealloc_benchmark(benchmark) -> None:
    benchmark(small_alloc_dealloc_benchmark)
