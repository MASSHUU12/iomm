def dynamic_array_benchmark():
    CAPACITY = 100_000_000

    arr = []

    for j in range(CAPACITY):
        arr.append(j)

    total = 0
    for v in arr:
        total += v

    arr = None


def test_dynamic_array_benchmark(benchmark) -> None:
    benchmark(dynamic_array_benchmark)


if __name__ == "__main__":
    dynamic_array_benchmark()
