#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
OUT_DIR="$ROOT/benchmark_results"
BIN_DIR="$ROOT"

mkdir -p "${OUT_DIR}"

for exe in "${BIN_DIR}"/test_*.zig; do
  name=$(basename "$exe")
  command=(zig test -O ReleaseFast -lc "${exe}")
  cmd="zig test -O ReleaseFast -lc \"$exe\""

  echo
  echo "=== Benchmarking ${name} ==="

  hyperfine \
    --warmup 3 \
    --runs 10 \
    --export-csv "${OUT_DIR}/${name}_time.csv" \
    --export-json "${OUT_DIR}/${name}_time.json" \
    -- \
    "${cmd}"

  /usr/bin/time \
    -f "${name},%e,%U,%S,%M" \
    -o "${OUT_DIR}/cpu_mem.csv" \
    --append \
    "${command[@]}"

  perf stat \
    -x"," \
    -e cycles,instructions,cache-references,cache-misses \
    -o "${OUT_DIR}/${name}_perf.csv" \
    -- \
    "${command[@]}"
done

echo
echo "All done. Results in ${OUT_DIR}/"
