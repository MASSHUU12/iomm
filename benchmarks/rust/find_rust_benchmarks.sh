#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
cd "$ROOT"

declare -a PATTERNS=(
  "dynamic_array_bench:target/release/deps/dynamic_array_bench-*"
  "large_alloc_bench:target/release/deps/large_alloc_bench-*"
  "large_buffer_reuse_bench:target/release/deps/large_buffer_reuse_bench-*"
  "linked_list_bench:target/release/deps/linked_list_bench-*"
  "parallel_alloc_bench:target/release/deps/parallel_alloc_bench-*"
  "shared_queue_bench:target/release/deps/shared_queue_bench-*"
  "short_lived_tasks_bench:target/release/deps/short_lived_tasks_bench-*"
  "small_alloc_bench:target/release/deps/small_alloc_bench-*"
  "small_buffer_reuse_bench:target/release/deps/small_buffer_reuse_bench-*"
)

echo "# Rust benchmark executables" > rust_benchmarks.txt

for pattern in "${PATTERNS[@]}"; do
    name="${pattern%%:*}"
    glob_pattern="${pattern#*:}"

    expanded_files=($glob_pattern)

    if [[ ${#expanded_files[@]} -gt 0 ]] && [[ -f "${expanded_files[0]}" ]] && [[ -x "${expanded_files[0]}" ]]; then
        executable="${expanded_files[0]}"
        echo "$name:$executable" >> rust_benchmarks.txt
        echo "Found benchmark: $name -> $executable" >&2
    else
        echo "Warning: No executable found for pattern: $glob_pattern" >&2
    fi
done

# echo "# Additional benchmark executables found automatically" >> rust_benchmarks.txt
# find target/release/deps -maxdepth 1 -type f -executable -name "*bench*" 2>/dev/null | while read -r exe; do
#   name=$(basename "$exe")
#   if ! grep -q ":$exe$" rust_benchmarks.txt; then
#     echo "$name:$exe" >> rust_benchmarks.txt
#     echo "Found additional benchmark: $name -> $exe" >&2
#   fi
# done

echo "Benchmark discovery complete. Found $(grep -c ":" rust_benchmarks.txt) benchmarks." >&2
