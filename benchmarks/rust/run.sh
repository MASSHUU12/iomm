#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=benchmark_results
export CRITERION_HOME="$OUT_DIR/criterion"

mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}/criterion"

rm -f "${OUT_DIR}"/**/*.{json,csv}

# Correctness tests:
#    cargo test -- --format=json > "${OUT_DIR}/tests.json"

cargo bench

while IFS= read -r -d $'\0' exe; do
  name=$(basename "$exe")

  echo "=== Benchmarking $name ==="

  # Hyperfine for distribution of runtimes
  hyperfine \
    --warmup 3 \
    --runs 10 \
    --export-csv "${OUT_DIR}/${name}_time.csv" \
    --export-json "${OUT_DIR}/${name}_time.json" \
    "$exe"

  # GNU time for CPU & peak memory
  /usr/bin/time \
    -f "$name,%e,%U,%S,%M" \
    -o "${OUT_DIR}/cpu_mem.csv" \
    --append \
    "$exe"

  # perf stat for hardware counters
  perf stat \
    -x"," \
    -e cycles,instructions,cache-references,cache-misses \
    -o "${OUT_DIR}/${name}_perf.csv" \
    -- \
    "$exe"
done < <(find target/release/deps -maxdepth 1 -type f -executable -name "*bench*" -print0)

echo "All done. Results in ${OUT_DIR}/"
