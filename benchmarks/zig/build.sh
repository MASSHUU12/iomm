#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="zig-test-bins"

mkdir -p "$OUT_DIR"

find . -type f -name "test_*.zig" | while read -r zigfile; do
    base=$(basename "$zigfile" .zig)
    outbin="$OUT_DIR/${base}_test"
    echo "Compiling $zigfile -> $outbin"
    zig test "$zigfile" -lc -O ReleaseFast --test-no-exec -femit-bin="$outbin"
done
