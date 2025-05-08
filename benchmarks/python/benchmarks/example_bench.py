def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def test_fib_correctness():
    # a simple correctness check for pytest
    assert fib(10) == 55


def test_fib_benchmark(benchmark):
    result = benchmark(fib, 20)
    assert result == 6765
