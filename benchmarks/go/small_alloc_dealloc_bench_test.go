package main

import (
	"testing"
)

func BenchmarkSmallAllocDealloc(b *testing.B) {
	const ITERS uint32 = 10_000_000
	const BYTES uint32 = 256

	for b.Loop() {
		for range ITERS {
			var _ = make([]byte, BYTES)
		}
	}
}
