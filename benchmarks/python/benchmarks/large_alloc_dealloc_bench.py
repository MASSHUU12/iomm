def large_alloc_dealloc_benchmark() -> None:
    ITERS = 100_000
    BYTES = 5 * 1_048_576

    for _ in range(0, ITERS):
        _ = bytearray(BYTES)

def test_large_alloc_dealloc_benchmark(benchmark) -> None:
    benchmark(large_alloc_dealloc_benchmark)
