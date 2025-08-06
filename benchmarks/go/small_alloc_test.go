package main

import (
	"testing"
)

var SSink []byte

func BenchmarkSmallAllocDealloc(b *testing.B) {
	const (
		ITERS = 10_000_000
		BYTES = 256
	)

	for b.Loop() {
		for range ITERS {
			SSink = make([]byte, BYTES)
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
		SSink = buf
	}
}
