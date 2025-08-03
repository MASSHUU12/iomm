def large_mem_reuse_alloc_benchmark() -> None:
    ITERS = 10_000
    BYTES = 5 * 1_048_576

    buf = bytearray(BYTES)
    for _ in range(ITERS):
        for i in range(len(buf)):
            buf[i] = 0

def test_large_mem_reuse_alloc_benchmark(benchmark) -> None:
    benchmark(large_mem_reuse_alloc_benchmark)
