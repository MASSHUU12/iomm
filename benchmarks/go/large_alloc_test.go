package main

import (
	"testing"
)

var Sink []byte

func BenchmarkLargeAllocDealloc(b *testing.B) {
	const (
		ITERS = 100_000
		BYTES = 5 * 1_048_576
	)

	for b.Loop() {
		for range ITERS {
			Sink = make([]byte, BYTES)
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
		Sink = buf
	}
}
