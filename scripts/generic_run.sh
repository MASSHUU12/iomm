#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:-benchmark.conf}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file '$CONFIG_FILE' not found!"
  echo "Usage: $0 [config_file]"
  echo "Create a configuration file or use the default 'benchmark.conf'"
  exit 1
fi

CONFIG_DIR="$(cd "$(dirname "$CONFIG_FILE")" && pwd)"
CONFIG_BASENAME="$(basename "$CONFIG_FILE")"

ROOT="$CONFIG_DIR"

cd "$ROOT"
source "$CONFIG_BASENAME"

# Set defaults if not specified in config
OUT_DIR="${OUT_DIR:-$ROOT/benchmark_results}"
TIMESTAMP="${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
SESSION_DIR="${SESSION_DIR:-${OUT_DIR}/session_${TIMESTAMP}}"
WARMUP_RUNS="${WARMUP_RUNS:-3}"
MEASUREMENT_RUNS="${MEASUREMENT_RUNS:-10}"
PERF_RECORD_FREQ="${PERF_RECORD_FREQ:-1000}"
ENABLE_PERF="${ENABLE_PERF:-true}"
ENABLE_HYPERFINE="${ENABLE_HYPERFINE:-true}"
ENABLE_TIME="${ENABLE_TIME:-true}"
ENABLE_SCHED="${ENABLE_SCHED:-true}"
CLEANUP_TEMP="${CLEANUP_TEMP:-true}"

mkdir -p "${SESSION_DIR}"

# Initialize CSV files
if [[ "$ENABLE_TIME" == "true" ]]; then
  echo "benchmark,elapsed_time,user_time,system_time,max_rss_kb,timestamp" > "${SESSION_DIR}/resource_usage.csv"
fi

if [[ "$ENABLE_SCHED" == "true" ]]; then
  echo "benchmark,run,mean_ms,median_ms,p95_ms,p99_ms,max_ms,switches,migrations" > "${SESSION_DIR}/latency_summary.csv"
fi

echo "=== Benchmark Runner ==="
echo "Session: ${TIMESTAMP}"
echo "Output directory: ${SESSION_DIR}"
echo "Configuration: ${CONFIG_FILE}"
echo "Working directory: ${CONFIG_DIR}"
echo

if [[ -n "${INIT_COMMAND:-}" ]]; then
  echo "=== Running one-time initialization ==="
  if eval "${INIT_COMMAND}"; then
    echo "Initialization completed successfully"
  else
    echo "Initialization failed"
    exit 1
  fi
  echo
fi

run_setup() {
  if [[ -n "${SETUP_COMMAND:-}" ]]; then
    echo "  Running setup: ${SETUP_COMMAND}"
    eval "${SETUP_COMMAND}"
  fi
}

run_cleanup() {
  if [[ -n "${CLEANUP_COMMAND:-}" ]]; then
    echo "  Running cleanup: ${CLEANUP_COMMAND}"
    eval "${CLEANUP_COMMAND}"
  fi
}

parse_command() {
  local cmd="$1"
  # Convert command string to array for proper execution
  eval "set -- $cmd"
  echo "$@"
}

run_benchmark() {
  local bench_name="$1"
  local bench_command="$2"
  eval "cmd_array=($bench_command)"

  echo "--- $bench_name ---"

  local bench_dir="${SESSION_DIR}/${bench_name}"
  mkdir -p "${bench_dir}"

  run_setup

  if [[ "$ENABLE_HYPERFINE" == "true" ]] && command -v hyperfine &> /dev/null; then
    echo "  Running timing analysis with hyperfine..."
    hyperfine \
      --warmup "$WARMUP_RUNS" \
      --runs "$MEASUREMENT_RUNS" \
      --export-csv "${bench_dir}/hyperfine_time.csv" \
      --export-json "${bench_dir}/hyperfine_time.json" \
      "$bench_command"
  fi

  if [[ "$ENABLE_TIME" == "true" ]]; then
    echo "  Measuring resource usage..."
    local time_output_file="${bench_dir}/time_stderr.log"

    if /usr/bin/time \
      -f "${bench_name},%e,%U,%S,%M,$(date +%s)" \
      -o "${SESSION_DIR}/resource_usage.csv" \
      --append \
      "${cmd_array[@]}" 2>"$time_output_file"; then
      echo "    Resource measurement completed successfully"
    else
      local exit_code=$?
      echo "    Resource measurement failed with exit code: $exit_code"
      if [[ -s "$time_output_file" ]]; then
        echo "    Error output:"
        cat "$time_output_file"
      fi
    fi
  fi

    if [[ "$ENABLE_PERF" == "true" ]] && command -v perf &> /dev/null; then
      echo "  Collecting performance counters..."
      local perf_stderr="${bench_dir}/perf_counters_stderr.log"

    if perf stat \
      -x"," \
      -e cycles,instructions,cache-references,cache-misses \
      -e L1-dcache-loads,L1-dcache-load-misses \
      -e LLC-loads,LLC-load-misses \
      -e branch-instructions,branch-misses \
      -o "${bench_dir}/perf_counters.csv" \
      -- "${cmd_array[@]}" 2>"$perf_stderr"; then
      echo "    Performance counters collected successfully"
    else
      echo "    Performance counters collection failed"
      if [[ -s "$perf_stderr" ]]; then
        echo "    Error output:"
        cat "$perf_stderr"
      fi
  fi

    echo "  Measuring context switches and migrations..."
    local perf_context_stderr="${bench_dir}/perf_context_stderr.log"

    if perf stat \
      -e 'context-switches' \
      -e 'cpu-migrations' \
      -e 'minor-faults' \
      -e 'major-faults' \
      -e 'cpu-clock' \
      -e 'task-clock' \
      -e 'page-faults' \
      -x"," \
      -o "${bench_dir}/context_stats.csv" \
      -- "${cmd_array[@]}" 2>"$perf_context_stderr"; then
      echo "    Context stats collected successfully"
    else
      echo "    Context stats collection failed"
      if [[ -s "$perf_context_stderr" ]]; then
        echo "    Error output:"
        cat "$perf_context_stderr"
      fi
    fi
  fi

  if [[ "$ENABLE_SCHED" == "true" ]] && command -v perf &> /dev/null; then
    echo "  Recording scheduling events..."
    local sched_stderr="${bench_dir}/perf_sched_stderr.log"

    if perf sched record \
      -e 'sched:sched_switch,sched:sched_wakeup,sched:sched_wakeup_new,sched:sched_migrate_task' \
      -m 512 \
      --output "${bench_dir}/sched.data" \
      -- "${cmd_array[@]}" 2>"$sched_stderr"; then
      echo "    Scheduling events recorded successfully"

      if [[ -f "${bench_dir}/sched.data" ]] && [[ -s "${bench_dir}/sched.data" ]]; then
        echo "  Analyzing scheduling latency..."
        perf sched latency \
          --input "${bench_dir}/sched.data" \
          --sort max \
          > "${bench_dir}/sched_latency.txt" 2>&1 || echo "    Latency analysis failed"

        perf sched map --input "${bench_dir}/sched.data" > "${bench_dir}/sched_map.txt" 2>&1 || echo "    Map generation failed"
        perf sched timehist --input "${bench_dir}/sched.data" > "${bench_dir}/sched_timeline.txt" 2>&1 || echo "    Timeline generation failed"
        echo "    Analyzing scheduling latency finished successfully"
      fi
    else
      echo "    Scheduling recording failed"
      if [[ -s "$sched_stderr" ]]; then
        echo "    Error output:"
        cat "$sched_stderr"
      fi
    fi
  fi

  if [[ -n "${CUSTOM_BENCHMARK_COMMAND:-}" ]]; then
    echo "  Running custom benchmark command..."

    export BENCH_NAME="$bench_name"
    export BENCH_COMMAND="$bench_command"
    export BENCH_DIR="$bench_dir"
    export ROOT="$ROOT"
    export SESSION_DIR="$SESSION_DIR"

    if eval "$CUSTOM_BENCHMARK_COMMAND"; then
      echo "    Custom benchmark completed successfully"
    else
      echo "    Custom benchmark command failed"
    fi

    unset BENCH_NAME BENCH_COMMAND BENCH_DIR
  fi

  run_cleanup

  echo "  Benchmark '$bench_name' completed"
}

# Main execution
echo "Starting benchmark execution..."

if declare -p BENCHMARKS 2>/dev/null | grep -q "declare -a"; then
  # BENCHMARKS is an array of "name:command" pairs
  for bench_spec in "${BENCHMARKS[@]}"; do
    if [[ "$bench_spec" == *":"* ]]; then
      bench_name="${bench_spec%%:*}"
      bench_command="${bench_spec#*:}"
    else
      bench_name="$bench_spec"
      bench_command="$bench_spec"
    fi
    run_benchmark "$bench_name" "$bench_command"
  done
elif [[ -n "${BENCHMARK_PATTERN:-}" ]]; then
  # Use pattern to find benchmark files
  for bench_file in $BENCHMARK_PATTERN; do
    if [[ -f "$bench_file" ]]; then
      bench_name=$(basename "$bench_file" .${bench_file##*.})
      bench_command="${BENCHMARK_COMMAND_TEMPLATE//\$FILE/$bench_file}"
      bench_command="${bench_command//\$NAME/$bench_name}"
      run_benchmark "$bench_name" "$bench_command"
    fi
  done
elif [[ -f "${BENCHMARK_LIST_FILE:-}" ]]; then
  # Read benchmarks from file
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

    if [[ "$line" == *":"* ]]; then
      bench_name="${line%%:*}"
      bench_command="${line#*:}"
    else
      bench_name="$line"
      bench_command="$line"
    fi
    run_benchmark "$bench_name" "$bench_command"
  done < "$BENCHMARK_LIST_FILE"
else
  echo "Error: No benchmark specification found!"
  echo "Please define BENCHMARKS array, BENCHMARK_PATTERN, or BENCHMARK_LIST_FILE in your config"
  exit 1
fi

if [[ -n "${ANALYSIS_COMMAND:-}" ]]; then
  echo "=== Running analysis ==="
  analysis_command="${ANALYSIS_COMMAND//\$SESSION_DIR/$SESSION_DIR}"
  if eval "$analysis_command"; then
    echo "Analysis completed successfully"
  else
    echo "Analysis failed"
  fi
fi

if [[ "$CLEANUP_TEMP" == "true" ]]; then
  find "$SESSION_DIR" -name "*_stderr.log" -size 0 -delete 2>/dev/null || true
  find "$SESSION_DIR" -name "sched_run_*.data" -delete 2>/dev/null || true
fi

echo
echo "All done. Results in ${SESSION_DIR}/"
echo "Configuration used: ${CONFIG_FILE}"
