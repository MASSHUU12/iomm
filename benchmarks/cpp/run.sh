#!/usr/bin/env bash
set -euo pipefail

for exe in build/bin/*; do
  echo "Running $exe"
  ./"$exe"
done
