def small_alloc_dealloc_benchmark() -> None:
  for _ in range(0, 1000000):
    _ = bytearray(100)

def test_small_alloc_dealloc_benchmark(benchmark) -> None:
  benchmark(small_alloc_dealloc_benchmark)
