package main

import (
	"runtime"
	"sync"
	"testing"
)

func BenchmarkParallelAlloc(b *testing.B) {
	const (
		TotalAllocs = 1_000_000_000
	)

	type Small struct {
		x, y, z int64
	}

	workers := runtime.GOMAXPROCS(0)
	perWorker := TotalAllocs / workers

	for b.Loop() {
		var wg sync.WaitGroup
		wg.Add(workers)

		for range workers {
			go func() {
				defer wg.Done()
				for range perWorker {
					_ = &Small{1, 2, 3}
				}
			}()
		}

		wg.Wait()
	}
}
