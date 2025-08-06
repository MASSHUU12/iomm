package main

import "testing"

var SinkInt int
var SinkIntSlice []int

func BenchmarkDynamicArray(b *testing.B) {
	const (
		CAPACITY = 1_000_000
	)

	for b.Loop() {
		arr := make([]int, 0)

		for j := range CAPACITY {
			arr = append(arr, j)
		}

		sum := 0
		for _, v := range arr {
			sum += v
		}

		SinkInt = sum
		SinkIntSlice = arr
	}
}

var SinkNode *Node

type Node struct {
	value int
	next  *Node
}

func BenchmarkLinkedList(b *testing.B) {
	const (
		ITERS = 10_000_000
		M     = 1_000_000
	)

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

		SinkInt = sum
		SinkNode = head
	}
}

var SinkTask *TaskData

type TaskData struct {
	a, b, c int
}

func BenchmarkShortLivedTasks(b *testing.B) {
	const (
		MTasks = 100_000_000
	)

	for b.Loop() {
		var last *TaskData
		for j := range MTasks {
			t := &TaskData{
				a: j,
				b: j * 2,
				c: j * 3,
			}
			last = t
		}
		SinkTask = last
	}
}
