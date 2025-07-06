package main

import (
	"testing"
)

func BenchmarkSmallAllocDealloc(b *testing.B) {
	const (
		ITERS = 10_000_000
		BYTES = 256
	)

	for b.Loop() {
		for range ITERS {
			var _ = make([]byte, BYTES)
		}
	}
}

func BenchmarkSmallReuse(b *testing.B) {
	const (
		ITERS = 10_000_000
		BYTES = 256
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
