#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[go] downloading modules…"
go mod download

echo "[go] compiling test binary…"
# -c: compile only, -o bench.test: name the binary
go test -c -o bench.test
