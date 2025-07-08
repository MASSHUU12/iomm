def reuse_alloc_benchmark() -> None:
    ITERS = 100_000
    BYTES = 256

    buf = bytearray(BYTES)
    for _ in range(ITERS):
        for i in range(len(buf)):
            buf[i] = 0

def test_reuse_alloc(benchmark) -> None:
    benchmark(reuse_alloc_benchmark)
