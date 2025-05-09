#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
OUT_DIR="$ROOT/benchmark_results"
BIN_DIR="$ROOT/zig-out/bin"

mkdir -p "${OUT_DIR}"

for exe in "${BIN_DIR}"/*; do
  name=$(basename "$exe")
  echo
  echo "=== Benchmarking ${name} ==="

  hyperfine \
    --warmup 3 \
    --runs 10 \
    --export-csv "${OUT_DIR}/${name}_time.csv" \
    --export-json "${OUT_DIR}/${name}_time.json" \
    "${exe}"

  /usr/bin/time \
    -f "${name},%e,%U,%S,%M" \
    -o "${OUT_DIR}/cpu_mem.csv" \
    --append \
    "${exe}"

  perf stat \
    -x"," \
    -e cycles,instructions,cache-references,cache-misses \
    -o "${OUT_DIR}/${name}_perf.csv" \
    -- \
    "${exe}"
done

echo
echo "All done. Results in ${OUT_DIR}/"
