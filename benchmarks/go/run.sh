#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
OUT_DIR="$ROOT/benchmark_results"
BIN="$ROOT/bench.test"

mkdir -p "${OUT_DIR}"

bench_names=()
while read -r line; do
  if [[ $line =~ ^Benchmark ]]; then
    bench_names+=("$line")
  fi
done < <("${BIN}" -test.list '^Benchmark')

for name in "${bench_names[@]}"; do
  echo
  echo "=== Benchmarking ${name} ==="

  hyperfine \
    --warmup 3 \
    --prepare "true" \
    --runs 10 \
    --export-csv "${OUT_DIR}/${name}_time.csv" \
    --export-json "${OUT_DIR}/${name}_time.json" \
    "${BIN} -test.bench=^${name}$ -test.run=^$" >/dev/null

  /usr/bin/time \
    -f "${name},%e,%U,%S,%M" \
    -o "${OUT_DIR}/cpu_mem.csv" \
    --append \
    -- "${BIN}" \
        -test.bench="^${name}$" \
        -test.run="^$"

  perf stat \
    -x"," \
    -e cycles,instructions,cache-references,cache-misses \
    -o "${OUT_DIR}/${name}_perf.csv" \
    -- \
    "${BIN}" \
      -test.bench="^${name}$" \
      -test.run="^$"
done

echo
echo "All done. Results in ${OUT_DIR}/"
