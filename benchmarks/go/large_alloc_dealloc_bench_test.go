package main

import (
	"testing"
)

func BenchmarkLargeAllocDealloc(b *testing.B) {
	const ITERS uint32 = 1_000_000
	const BYTES uint32 = 5 * 1_048_576

	for b.Loop() {
		var local = make([]*small, ITERS)
		for j := range ITERS {
			local[j] = &small{
				data: make([]byte, BYTES),
			}
		}
		local = nil
	}
}
