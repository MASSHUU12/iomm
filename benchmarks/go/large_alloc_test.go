package main

import (
	"testing"
)

func BenchmarkLargeAllocDealloc(b *testing.B) {
	const (
		ITERS = 100_000
		BYTES = 5 * 1_048_576
	)

	for b.Loop() {
		for range ITERS {
			var _ = make([]byte, BYTES)
		}
	}
}

func BenchmarkLargeReuse(b *testing.B) {
	const (
		ITERS = 100_000
		BYTES = 5 * 1_048_576
	)

	buf := make([]byte, BYTES)

	for b.Loop() {
		for range ITERS {
			for k := range buf {
				buf[k] = 0
			}
		}
	}
}

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
	}
}
