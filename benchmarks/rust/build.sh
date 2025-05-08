#!/usr/bin/env bash
set -euo pipefail

cargo fetch
cargo build --release
cargo bench --no-run
