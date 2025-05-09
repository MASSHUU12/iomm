#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=benchmark_results
mkdir -p "${OUT_DIR}"
rm -f "${OUT_DIR}"/*.{json,csv,xml}

source .venv/bin/activate

# echo "=== Running pytest unit tests ==="
# pytest tests \
#   --maxfail=1 --disable-warnings -q \
#   --junitxml="${OUT_DIR}/tests.xml"

echo "=== Running pytest-benchmark benchmarks ==="
for bench in benchmarks/*.py; do
  name=$(basename "$bench" .py)
  echo "--- Benchmarking $name ---"

  pytest "$bench" \
    --benchmark-only \
    --benchmark-json="${OUT_DIR}/${name}_bench.json" \
    --benchmark-warmup=true \
    --benchmark-warmup-iterations=10

  hyperfine \
    --warmup 3 \
    --runs 10 \
    --export-csv "${OUT_DIR}/${name}_time.csv" \
    --export-json "${OUT_DIR}/${name}_time.json" \
    "python $bench"

  /usr/bin/time \
    -f "$name,%e,%U,%S,%M" \
    -o "${OUT_DIR}/cpu_mem.csv" \
    --append \
    python "$bench"

  perf stat \
    -x"," \
    -e cycles,instructions,cache-references,cache-misses \
    -o "${OUT_DIR}/${name}_perf.csv" \
    -- \
    python "$bench"
done

echo "All done. Results in ${OUT_DIR}/"
