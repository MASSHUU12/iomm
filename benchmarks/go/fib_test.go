package main

import "testing"

func fibonacci(n int) int {
    if n < 2 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}

func BenchmarkFib10(b *testing.B) {
    for i := 0; i < b.N; i++ {
        _ = fibonacci(10)
    }
}

func BenchmarkFib20(b *testing.B) {
    for i := 0; i < b.N; i++ {
        _ = fibonacci(20)
    }
}
