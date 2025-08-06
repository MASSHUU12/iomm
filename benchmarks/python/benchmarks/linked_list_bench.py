class Node:
    __slots__ = ('value', 'next')
    def __init__(self, value, nxt):
        self.value = value
        self.next = nxt


def linked_list_benchmark():
    M = 1_000_000

    head = None
    for j in range(M):
        head = Node(j, head)

    total = 0
    cur = head
    while cur is not None:
        total += cur.value
        cur = cur.next


def test_linked_list_benchmark(benchmark) -> None:
    benchmark(linked_list_benchmark)


if __name__ == "__main__":
    linked_list_benchmark()
