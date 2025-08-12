#!/usr/bin/env bash
set -euo pipefail

LANGS=( go python rust zig )
ROOT=$(cd "$(dirname "$0")" && pwd)
DIR="benchmarks"

usage() {
  echo "Usage: $0 [lang1 lang2 ...]"
  echo "If no languages are provided, all languages will be processed."
  echo "Supported languages: ${LANGS[*]}"
  exit 1
}

check_dependencies() {
  local deps=("hyperfine" "perf" "/usr/bin/time")
  for dep in "${deps[@]}"; do
    if ! command -v "$dep" &> /dev/null; then
      echo "Error: $dep is required but not installed"
      exit 1
    fi
  done

  if [ "$EUID" -ne 0 ]; then
    echo "Please run as root,"
    exit 1
  fi
}

check_dependencies

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
fi

if [ "$#" -eq 0 ]; then
  SELECTED_LANGS=( "${LANGS[@]}" )
else
  declare -A SUPPORTED
  for lang in "${LANGS[@]}"; do
    SUPPORTED["$lang"]=1
  done

  SELECTED_LANGS=()
  for arg in "$@"; do
    if [[ -n "${SUPPORTED[$arg]:-}" ]]; then
      SELECTED_LANGS+=( "$arg" )
    else
      echo "Error: Unsupported language '$arg'"
      usage
    fi
  done
fi

for lang in "${SELECTED_LANGS[@]}"; do
  echo
  echo "[$lang] building..."
  ( cd "$ROOT/$DIR/$lang" && ./build.sh )

  echo "[$lang] running..."
  ( ./scripts/generic_run.sh "$ROOT/$DIR/$lang/benchmark.conf" )
done

echo
echo "Benchmarks complete. Outputs in results/"
