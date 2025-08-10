class TaskData:
    __slots__ = ('a','b','c')
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c


def short_lived_tasks_benchmark():
    MTasks = 1_000_000_000

    for j in range(MTasks):
        t = TaskData(j, j*2, j*3)
        _ = t.a + t.b + t.c


def test_short_lived_tasks_benchmark(benchmark) -> None:
    benchmark(short_lived_tasks_benchmark)


if __name__ == "__main__":
    short_lived_tasks_benchmark()
