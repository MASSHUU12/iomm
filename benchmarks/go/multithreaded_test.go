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

type queue struct {
	mu    sync.Mutex
	cond  *sync.Cond
	items []int
}

func newQueue() *queue {
	q := &queue{items: make([]int, 0, 1024)}
	q.cond = sync.NewCond(&q.mu)
	return q
}

func (q *queue) push(v int) {
	q.mu.Lock()
	q.items = append(q.items, v)
	q.cond.Signal()
	q.mu.Unlock()
}

func (q *queue) pop() int {
	q.mu.Lock()
	for len(q.items) == 0 {
		q.cond.Wait()
	}
	v := q.items[0]
	q.items[0] = 0
	q.items = q.items[1:]
	q.mu.Unlock()
	return v
}

func BenchmarkSharedQueue(b *testing.B) {
	const (
		TotalOps = 1_000_000
	)

	workers := runtime.GOMAXPROCS(0)
	producers := workers / 2
	consumers := workers - producers

	pushesPerProducer := TotalOps / 2 / producers
	popsPerConsumer := TotalOps / 2 / consumers

	for b.Loop() {
		q := newQueue()
		var wg sync.WaitGroup
		wg.Add(producers + consumers)

		// start producers
		for p := range producers {
			go func(seed int) {
				defer wg.Done()
				base := seed * pushesPerProducer
				for i := range pushesPerProducer {
					q.push(base + i)
				}
			}(p)
		}

		// start consumers
		for range consumers {
			go func() {
				defer wg.Done()
				for range popsPerConsumer {
					_ = q.pop()
				}
			}()
		}

		wg.Wait()
	}
}
