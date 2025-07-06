package main

import (
	"testing"
)

type small struct {
	data []byte
}

func BenchmarkSmallAllocDealloc(b *testing.B) {
	const ITERS uint32 = 1_000_000
	const BYTES uint32 = 128

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
