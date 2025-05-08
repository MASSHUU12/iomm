#!/usr/bin/env bash

set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ROOT/results"

LANGS=( rust )
DIR="benchmarks"

for lang in "${LANGS[@]}"; do
  echo
  echo "[$lang] building..."
  ( cd "$ROOT/$DIR/$lang" && ./build.sh )

  echo "[$lang] running..."
  ( cd "$ROOT/$DIR/$lang" && ./run.sh )
done

echo
echo "All benchmarks complete. Outputs in results/"
