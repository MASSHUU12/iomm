#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
ROOT=$(pwd)
OUT_BIN_DIR="$ROOT/bin"

mkdir -p "${OUT_BIN_DIR}"

echo "[go] downloading modules..."
go mod download

echo "[go] compiling test binary…"
go test -c -o "${OUT_BIN_DIR}/bench.test" ./...

echo "[go] built ${OUT_BIN_DIR}/bench.test"
