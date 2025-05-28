ITERATIONS = 10_000
BLOCK_SIZE = 10 * 1_048_576

def reuse_alloc_benchmark() -> None:
    buf = bytearray(BLOCK_SIZE)
    for _ in range(ITERATIONS):
        for i in range(len(buf)):
            buf[i] = 0

def test_reuse_alloc(benchmark) -> None:
    benchmark(reuse_alloc_benchmark)
