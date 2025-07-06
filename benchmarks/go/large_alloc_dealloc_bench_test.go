package main

import (
	"testing"
)

func BenchmarkLargeAllocDealloc(b *testing.B) {
	const ITERS uint32 = 10_000_000
	const BYTES uint32 = 5 * 1_048_576

	for b.Loop() {
		for range ITERS {
			var _ = make([]byte, BYTES)
		}
	}
}
