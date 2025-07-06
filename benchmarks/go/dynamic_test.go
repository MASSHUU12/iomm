package main

import "testing"

func BenchmarkDynamicArray(b *testing.B) {
	const (
		CAPACITY = 1_000_000
	)

	for b.Loop() {
		arr := make([]int, 0, CAPACITY)

		for j := range CAPACITY {
			arr = append(arr, j)
		}

		sum := 0
		for _, v := range arr {
			sum += v
		}

		arr = nil
	}
}

func BenchmarkLinkedList(b *testing.B) {
	const (
		ITERS = 10_000_000
		M     = 1_000_000
	)

	type Node struct {
		value int
		next  *Node
	}

	for b.Loop() {
		var head *Node
		for j := range M {
			n := new(Node)
			n.value = j
			n.next = head
			head = n
		}

		sum := 0
		for cur := head; cur != nil; cur = cur.next {
			sum += cur.value
		}
		_ = sum

		head = nil
	}
}

func BenchmarkShortLivedTasks(b *testing.B) {
	const (
		MTasks = 100_000_000
	)

	type TaskData struct {
		a, b, c int
	}

	for b.Loop() {
		for j := range MTasks {
			t := &TaskData{
				a: j,
				b: j * 2,
				c: j * 3,
			}

			sum := t.a + t.b + t.c
			_ = sum
		}
	}
}
