def dynamic_array_benchmark():
    CAPACITY = 1_000_000

    arr = []

    for j in range(CAPACITY):
        arr.append(j)

    total = 0
    for v in arr:
        total += v

    arr = None


def test_dynamic_array_benchmark(benchmark) -> None:
    benchmark(dynamic_array_benchmark)
