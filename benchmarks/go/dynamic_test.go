package main

import "testing"

var DSinkInt int
var DSinkIntSlice []int
var DSinkNode *Node
var DSinkTask *TaskData

func BenchmarkDynamicArray(b *testing.B) {
	const (
		CAPACITY = 100_000_000
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

		DSinkInt = sum
		DSinkIntSlice = arr
	}
}

type Node struct {
	value int
	next  *Node
}

func BenchmarkLinkedList(b *testing.B) {
	const (
		M = 100_000_000
	)

	for b.Loop() {
		var head *Node
		for j := range M {
			n := &Node{value: j, next: head}
			head = n
		}

		sum := 0
		for cur := head; cur != nil; cur = cur.next {
			sum += cur.value
		}

		DSinkInt = sum
		DSinkNode = head
	}
}

type TaskData struct {
	a, b, c int
}

func BenchmarkShortLivedTasks(b *testing.B) {
	const (
		MTasks = 1_000_000_000
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
		DSinkTask = last
	}
}
