// basic_bench.cpp
#include <chrono>
#include <iostream>
#include <vector>

// volatile global sinks to prevent optimizations
volatile int sink_int;
volatile long long sink_ll;

void print_duration(const char *label,
                    const std::chrono::high_resolution_clock::time_point &start,
                    const std::chrono::high_resolution_clock::time_point &end) {
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
  std::cout << label << ": " << ms << " ms\n";
}

int main() {
  using clock = std::chrono::high_resolution_clock;

  // Test 1: push_back N ints into a vector
  {
    const size_t N = 100000;
    const auto t0 = clock::now();
    std::vector<int> v;
    v.reserve(N);
    for (size_t i = 0; i < N; ++i) {
      v.push_back(static_cast<int>(i));
    }
    const auto t1 = clock::now();
    print_duration("Vector push_back 100 000 elems", t0, t1);

    if (!v.empty()) {
      sink_int = v[0];
    }
  }

  // Test 2: sum integers 0..N-1 in a loop
  {
    const size_t N = 1000000;
    const auto t0 = clock::now();
    long long sum = 0;
    for (size_t i = 0; i < N; ++i) {
      sum += static_cast<long long>(i);
    }
    const auto t1 = clock::now();
    print_duration("Sum 1 000 000 ints", t0, t1);

    std::cout << "  (checksum = " << sum << ")\n";
    sink_ll = sum;
  }

  return 0;
}
