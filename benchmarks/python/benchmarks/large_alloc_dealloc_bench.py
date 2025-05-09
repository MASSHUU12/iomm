def large_alloc_dealloc_benchmark() -> None:
  for _ in range(0, 10000):
    _ = bytearray(10 * 1_048_576)

def test_large_alloc_dealloc_benchmark(benchmark) -> None:
  benchmark(large_alloc_dealloc_benchmark)
