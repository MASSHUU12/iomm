#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
OUT_DIR="$ROOT/benchmark_results"
BIN="$ROOT/bin/bench.test"

mkdir -p "${OUT_DIR}"
rm -rf "${OUT_DIR}"/*.{csv,json}

BENCHES=(
  BenchmarkSmallAllocDealloc
  BenchmarkLargeAllocDealloc
  BenchmarkSmallReuse
  BenchmarkLargeReuse
  BenchmarkDynamicArray
)

echo "=== Running Go benchmarks ==="

for bench in "${BENCHES[@]}"; do
  echo "--- $bench ---"

  hyperfine \
    --warmup 3 \
    --runs 10 \
    --export-csv "${OUT_DIR}/${bench}_time.csv" \
    --export-json "${OUT_DIR}/${bench}_time.json" \
    "\"${BIN}\" -test.bench=^${bench}$ -test.benchtime=1x"

  /usr/bin/time \
    -f "${bench},%e,%U,%S,%M" \
    -o "${OUT_DIR}/cpu_mem.csv" \
    --append \
    "${BIN}" -test.bench=^${bench}$ -test.benchtime=1x

  perf stat \
    -x"," \
    -e cycles,instructions,cache-references,cache-misses \
    -o "${OUT_DIR}/${bench}_perf.csv" \
    -- \
    "${BIN}" -test.bench=^${bench}$ -test.benchtime=1x

done

echo
echo "All done. Results in ${OUT_DIR}/"
