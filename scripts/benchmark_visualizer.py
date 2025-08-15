#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from typing import Dict

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BenchmarkVisualizer:
    def __init__(self, session_dir: str, output_dir: Optional[str] = None):
        self.session_dir = Path(session_dir)
        self.output_dir = Path(output_dir) if output_dir else self.session_dir / "graphs"
        self.output_dir.mkdir(exist_ok=True)

        self.hyperfine_data = {}
        self.resource_data = None
        self.perf_counters = {}
        self.context_stats = {}
        self.sched_latency = {}
        self.sched_map = {}
        self.sched_timeline = {}

        self._load_data()


    def _load_data(self):
        print(f"Loading data from: {self.session_dir}")

        self._load_hyperfine_data()
        self._load_resource_data()
        self._load_perf_data()
        self._load_sched_data()

        print(f"Data loading complete. Graphs will be saved to: {self.output_dir}")


    def _load_hyperfine_data(self):
        for bench_dir in self.session_dir.iterdir():
            if bench_dir.is_dir():
                bench_name = bench_dir.name

                csv_file = bench_dir / "hyperfine_time.csv"
                if csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file)
                        self.hyperfine_data[bench_name] = df
                    except Exception as e:
                        print(f"Warning: Failed to load {csv_file}: {e}")

                json_file = bench_dir / "hyperfine_time.json"
                if json_file.exists():
                    try:
                        with open(json_file, 'r') as f:
                            json_data = json.load(f)
                            if bench_name not in self.hyperfine_data:
                                self.hyperfine_data[bench_name] = {}
                            self.hyperfine_data[bench_name]['json'] = json_data
                    except Exception as e:
                        print(f"Warning: Failed to load {json_file}: {e}")


    def _load_resource_data(self):
        resource_file = self.session_dir / "resource_usage.csv"
        if resource_file.exists():
            try:
                self.resource_data = pd.read_csv(resource_file)
                if 'timestamp' in self.resource_data.columns:
                    self.resource_data['datetime'] = pd.to_datetime(
                        self.resource_data['timestamp'], unit='s'
                    )
            except Exception as e:
                print(f"Warning: Failed to load {resource_file}: {e}")


    def _parse_perf_csv(self, file_path: Path) -> pd.DataFrame:
        rows = []

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()

                    if line.startswith('#') or not line:
                        continue

                    fields = [field.strip() for field in line.split(',')]

                    if len(fields) >= 3:
                        row = {
                            'value': fields[0],
                            'unit': fields[1] if fields[1] else '',
                            'event': fields[2],
                            'runtime': fields[3] if len(fields) > 3 else '',
                            'percentage': fields[4] if len(fields) > 4 else '',
                            'additional_info': ','.join(fields[5:]) if len(fields) > 5 else ''
                        }
                        rows.append(row)

        except Exception as e:
            print(f"Error parsing perf CSV {file_path}: {e}")
            return pd.DataFrame()

        return pd.DataFrame(rows)


    def _load_perf_data(self):
        for bench_dir in self.session_dir.iterdir():
            if bench_dir.is_dir():
                bench_name = bench_dir.name

                perf_file = bench_dir / "perf_counters.csv"
                if perf_file.exists():
                    try:
                        df = self._parse_perf_csv(perf_file)
                        if not df.empty:
                            self.perf_counters[bench_name] = df
                    except Exception as e:
                        print(f"Warning: Failed to load {perf_file}: {e}")

                context_file = bench_dir / "context_stats.csv"
                if context_file.exists():
                    try:
                        df = self._parse_perf_csv(context_file)
                        if not df.empty:
                            self.context_stats[bench_name] = df
                    except Exception as e:
                        print(f"Warning: Failed to load {context_file}: {e}")


    def _load_sched_data(self):
        for bench_dir in self.session_dir.iterdir():
            if bench_dir.is_dir():
                bench_name = bench_dir.name

                sched_latency_file = bench_dir / "sched_latency.txt"
                if sched_latency_file.exists():
                    try:
                        latency_data = self._parse_sched_latency(sched_latency_file)
                        if latency_data is not None and not latency_data.empty:
                            self.sched_latency[bench_name] = latency_data
                            print(f"✓ Loaded scheduling latency data for {bench_name}: {len(latency_data)} tasks")
                    except Exception as e:
                        print(f"Warning: Failed to parse {sched_latency_file}: {e}")

                sched_map_file = bench_dir / "sched_map.txt"
                if sched_map_file.exists():
                    try:
                        map_data = self._parse_sched_map(sched_map_file)
                        if map_data is not None:
                            self.sched_map[bench_name] = map_data
                            print(f"✓ Loaded scheduling map data for {bench_name}")
                    except Exception as e:
                        print(f"Warning: Failed to parse {sched_map_file}: {e}")

                sched_timeline_file = bench_dir / "sched_timeline.txt"
                if sched_timeline_file.exists():
                    try:
                        timeline_data = self._parse_sched_timeline(sched_timeline_file)
                        if timeline_data is not None and not timeline_data.empty:
                            self.sched_timeline[bench_name] = timeline_data
                            print(f"✓ Loaded scheduling timeline data for {bench_name}: {len(timeline_data)} entries")
                    except Exception as e:
                        print(f"Warning: Failed to parse {sched_timeline_file}: {e}")


    def _parse_sched_latency(self, file_path: Path) -> Optional[pd.DataFrame]:
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            lines = [line for line in content.split('\n')
                     if line.strip() and not line.strip().startswith('-')]

            header_idx = -1
            for i, line in enumerate(lines):
                if 'Task' in line and 'Runtime ms' in line and 'Count' in line:
                    header_idx = i
                    break

            if header_idx == -1:
                print(f"No header found in {file_path}")
                return None

            rows = []
            for i in range(header_idx + 1, len(lines)):
                line = lines[i].strip()

                if line.startswith('TOTAL:'):
                    break

                parts = [part.strip() for part in line.split('|')]

                if len(parts) >= 7:
                    task_name = parts[0]
                    runtime_str = parts[1]
                    count_str = parts[2]
                    avg_delay_str = parts[3]
                    max_delay_str = parts[4]
                    max_start_str = parts[5]
                    max_end_str = parts[6]

                    runtime = self._extract_number_with_unit(runtime_str)
                    count = int(count_str.strip())
                    avg_delay = self._extract_number_with_prefix(avg_delay_str, 'avg:')
                    max_delay = self._extract_number_with_prefix(max_delay_str, 'max:')
                    max_start = self._extract_number_with_prefix(max_start_str, 'max start:')
                    max_end = self._extract_number_with_prefix(max_end_str, 'max end:')

                    rows.append({
                        'task': task_name,
                        'runtime_ms': runtime,
                        'switches': count,
                        'avg_delay_ms': avg_delay,
                        'max_delay_ms': max_delay,
                        'max_start_s': max_start,
                        'max_end_s': max_end
                    })

            if rows:
                return pd.DataFrame(rows)
            else:
                print(f"No valid data rows found in {file_path}")
                return None

        except Exception as e:
            print(f"Error parsing sched latency file {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None


    def _parse_sched_map(self, file_path: Path) -> Optional[Dict[str, str]]:
        try:
            task_map = {}
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Looking for lines like: *A0 8469.477325 secs A0 => migration/0:19
                    match = re.search(r'\s+([A-Z][0-9]+)\s+=>\s+(.+)$', line)
                    if match:
                        task_id = match.group(1)
                        task_name = match.group(2)
                        task_map[task_id] = task_name

            return task_map if task_map else None

        except Exception as e:
            print(f"Error parsing sched map file {file_path}: {e}")
            return None


    def _parse_sched_timeline(self, file_path: Path) -> Optional[pd.DataFrame]:
        try:
            rows = []
            header_found = False

            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()

                    if not line or line.startswith('Samples'):
                        continue

                    if 'time' in line and 'cpu' in line and 'task name' in line:
                        header_found = True
                        continue

                    if line.startswith('----'):
                        continue

                    if header_found:
                        try:
                            time_str = line[:15].strip()
                            cpu_str = line[15:23].strip()
                            task_info = line[23:52].strip()
                            wait_time = line[52:63].strip()
                            sch_delay = line[63:74].strip()
                            run_time = line[74:].strip()

                            cpu_match = re.search(r'\[(\d+)\]', cpu_str)
                            cpu = int(cpu_match.group(1)) if cpu_match else -1

                            task_pid_match = re.search(r'([^\[]+)\[(\d+)\]', task_info)
                            if task_pid_match:
                                task_name = task_pid_match.group(1).strip()
                                task_pid = int(task_pid_match.group(2))
                            else:
                                task_name = task_info
                                task_pid = -1

                            timestamp = float(time_str)
                            wait_time_ms = float(wait_time) if wait_time else 0.0
                            sch_delay_ms = float(sch_delay) if sch_delay else 0.0
                            run_time_ms = float(run_time) if run_time else 0.0

                            rows.append({
                                'timestamp': timestamp,
                                'cpu': cpu,
                                'task_name': task_name,
                                'pid': task_pid,
                                'wait_time_ms': wait_time_ms,
                                'sched_delay_ms': sch_delay_ms,
                                'run_time_ms': run_time_ms
                            })
                        except Exception:
                            pass

            return pd.DataFrame(rows) if rows else None

        except Exception as e:
            print(f"Error parsing sched timeline file {file_path}: {e}")
            return None


    def _extract_number_with_unit(self, text: str) -> Optional[float]:
        try:
            match = re.search(r'([\d\.]+)\s*[a-zA-Z]+', text)
            if match:
                return float(match.group(1))
            return None
        except:
            return None


    def _extract_number_with_prefix(self, text: str, prefix: str = '') -> Optional[float]:
        try:
            if prefix and prefix in text:
                start_pos = text.find(prefix) + len(prefix)
                remaining_text = text[start_pos:].strip()
                match = re.search(r'([\d\.]+)\s*[a-zA-Z]+', remaining_text)
                if match:
                    return float(match.group(1))
            return None
        except:
            return None


    def _extract_number(self, text: str) -> Optional[float]:
        try:
            cleaned = re.sub(r'[^\d\.-]', '', text.strip())
            return float(cleaned) if cleaned else None
        except:
            return None


    def _get_perf_metric_value(self, data: pd.DataFrame, metric_name: str) -> Optional[float]:
        try:
            metric_row = data[data['event'].str.contains(metric_name, na=False, case=False)]
            if not metric_row.empty:
                value_str = metric_row.iloc[0]['value']
                cleaned_value = value_str.replace(',', '')
                return float(cleaned_value) if cleaned_value else None
        except (ValueError, AttributeError, IndexError):
            pass
        return None


    def plot_execution_times(self):
        if not self.hyperfine_data:
            print("No hyperfine timing data available for plotting.")
            return

        benchmarks = []
        means = []
        stds = []

        for bench_name, data in self.hyperfine_data.items():
            if isinstance(data, dict) and 'json' in data:
                results = data['json'].get('results', [])
                if results:
                    result = results[0]  # First (and usually only) result
                    benchmarks.append(bench_name)
                    means.append(result.get('mean', 0))
                    stds.append(result.get('stddev', 0))
            elif isinstance(data, pd.DataFrame) and 'mean' in data.columns:
                benchmarks.append(bench_name)
                means.append(data['mean'].iloc[0])
                stds.append(data['stddev'].iloc[0] if 'stddev' in data.columns else 0)

        if not benchmarks:
            print("No timing data found for execution time plots.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        bars = ax1.bar(benchmarks, means, yerr=stds, capsize=5, alpha=0.7)
        ax1.set_xlabel('Benchmark')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Benchmark Execution Times')
        ax1.tick_params(axis='x', rotation=45)

        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{mean:.3f}s', ha='center', va='bottom', fontsize=8)

        distributions = []
        labels = []
        for bench_name, data in self.hyperfine_data.items():
            if isinstance(data, pd.DataFrame) and 'times' in data.columns:
                distributions.append(data['times'].values)
                labels.append(bench_name)

        if distributions:
            ax2.boxplot(distributions, labels=labels)
            ax2.set_xlabel('Benchmark')
            ax2.set_ylabel('Execution Time (seconds)')
            ax2.set_title('Execution Time Distributions')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No distribution data available',
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('Execution Time Distributions (No Data)')

        plt.tight_layout()
        plt.savefig(self.output_dir / "execution_times.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated execution_times.png")


    def plot_resource_usage(self):
        if self.resource_data is None or self.resource_data.empty:
            print("No resource usage data available for plotting.")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        benchmarks = self.resource_data['benchmark'].values

        user_time = self.resource_data['user_time'].values
        system_time = self.resource_data['system_time'].values

        x = np.arange(len(benchmarks))
        width = 0.35

        ax1.bar(x - width/2, user_time, width, label='User Time', alpha=0.7)
        ax1.bar(x + width/2, system_time, width, label='System Time', alpha=0.7)
        ax1.set_xlabel('Benchmark')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('CPU Time Breakdown')
        ax1.set_xticks(x)
        ax1.set_xticklabels(benchmarks, rotation=45)
        ax1.legend()

        memory_mb = self.resource_data['max_rss_kb'].values / 1024
        bars = ax2.bar(benchmarks, memory_mb, alpha=0.7)
        ax2.set_xlabel('Benchmark')
        ax2.set_ylabel('Peak Memory Usage (MB)')
        ax2.set_title('Peak Memory Usage')
        ax2.tick_params(axis='x', rotation=45)

        for bar, mem in zip(bars, memory_mb):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mem:.1f}MB', ha='center', va='bottom', fontsize=8)

        total_time = self.resource_data['elapsed_time'].values
        cpu_time = user_time + system_time
        cpu_efficiency = (cpu_time / total_time) * 100

        ax3.bar(benchmarks, cpu_efficiency, alpha=0.7)
        ax3.set_xlabel('Benchmark')
        ax3.set_ylabel('CPU Efficiency (%)')
        ax3.set_title('CPU Utilization Efficiency')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 100)

        if 'datetime' in self.resource_data.columns:
            ax4.plot(self.resource_data['datetime'], total_time, 'o-', label='Elapsed Time')
            ax4.plot(self.resource_data['datetime'], cpu_time, 's-', label='CPU Time')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Time (seconds)')
            ax4.set_title('Execution Timeline')
            ax4.legend()
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No timestamp data available',
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Execution Timeline (No Data)')

        plt.tight_layout()
        plt.savefig(self.output_dir / "resource_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated resource_usage.png")


    def plot_performance_counters(self):
        if not self.perf_counters:
            print("No performance counter data available for plotting.")
            return

        metrics = ['cycles', 'instructions', 'cache-references', 'cache-misses',
                  'branch-instructions', 'branch-misses', 'L1-dcache-loads', 'L1-dcache-load-misses']

        n_metrics = len(metrics)
        cols = 3
        rows = (n_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
        if rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()

        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break

            ax = axes[i]
            benchmark_names = []
            values = []

            for bench_name, data in self.perf_counters.items():
                value = self._get_perf_metric_value(data, metric)
                if value is not None:
                    benchmark_names.append(bench_name)
                    values.append(value)

            if benchmark_names and values:
                bars = axes[i].bar(benchmark_names, values, alpha=0.7)
                axes[i].set_xlabel('Benchmark')
                axes[i].set_ylabel('Count')
                axes[i].set_title(f'{metric.replace("-", " ").title()}')
                axes[i].tick_params(axis='x', rotation=45)

                if len(values) <= 5:
                    for bar, val in zip(bars, values):
                        height = bar.get_height()
                        if val >= 1e9:
                            label = f'{val/1e9:.1f}B'
                        elif val >= 1e6:
                            label = f'{val/1e6:.1f}M'
                        elif val >= 1e3:
                            label = f'{val/1e3:.1f}K'
                        else:
                            label = f'{val:.0f}'
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               label, ha='center', va='bottom', fontsize=8)
            else:
                axes[i].text(0.5, 0.5, f'No {metric} data',
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f'{metric.replace("-", " ").title()} (No Data)')

        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_counters.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated performance_counters.png")


    def plot_context_switches(self):
        if not self.context_stats:
            print("No context switch data available for plotting.")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        benchmarks = []
        context_switches = []
        cpu_migrations = []
        page_faults = []
        task_clock = []

        for bench_name, data in self.context_stats.items():
            benchmarks.append(bench_name)

            context_switches.append(self._get_perf_metric_value(data, 'context-switches') or 0)
            cpu_migrations.append(self._get_perf_metric_value(data, 'cpu-migrations') or 0)
            page_faults.append(self._get_perf_metric_value(data, 'page-faults') or 0)

            tc_value = self._get_perf_metric_value(data, 'task-clock')
            task_clock.append(tc_value or 0)

        if not benchmarks:
            print("No context switch data found for plotting.")
            return

        bars1 = ax1.bar(benchmarks, context_switches, alpha=0.7)
        ax1.set_xlabel('Benchmark')
        ax1.set_ylabel('Context Switches')
        ax1.set_title('Context Switches per Benchmark')
        ax1.tick_params(axis='x', rotation=45)

        bars2 = ax2.bar(benchmarks, cpu_migrations, alpha=0.7, color='orange')
        ax2.set_xlabel('Benchmark')
        ax2.set_ylabel('CPU Migrations')
        ax2.set_title('CPU Migrations per Benchmark')
        ax2.tick_params(axis='x', rotation=45)

        bars3 = ax3.bar(benchmarks, page_faults, alpha=0.7, color='red')
        ax3.set_xlabel('Benchmark')
        ax3.set_ylabel('Page Faults')
        ax3.set_title('Page Faults per Benchmark')
        ax3.tick_params(axis='x', rotation=45)

        if any(tc > 0 for tc in task_clock):
            cs_per_sec = [cs / (tc / 1000) if tc > 0 else 0
                         for cs, tc in zip(context_switches, task_clock)]
            ax4.bar(benchmarks, cs_per_sec, alpha=0.7, color='green')
            ax4.set_xlabel('Benchmark')
            ax4.set_ylabel('Context Switches/sec')
            ax4.set_title('Context Switch Rate')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No timing data for rate calculation',
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Context Switch Rate (No Data)')

        plt.tight_layout()
        plt.savefig(self.output_dir / "context_switches.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated context_switches.png")


    def plot_scheduling_latency(self):
        if not self.sched_latency:
            print("No scheduling latency data available for plotting.")
            return

        num_benchmarks = len(self.sched_latency)
        cols = min(2, num_benchmarks)
        rows = (num_benchmarks + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        for i, (bench_name, data) in enumerate(self.sched_latency.items()):
            if i >= len(axes):
                break

            ax = axes[i]

            if not data.empty and 'avg_delay_ms' in data.columns and 'max_delay_ms' in data.columns:
                sizes = data['switches'].values
                sizes = 20 + (sizes - sizes.min()) / (sizes.max() - sizes.min() + 0.1) * 100

                scatter = ax.scatter(data['avg_delay_ms'], data['max_delay_ms'],
                                   s=sizes, alpha=0.6, c=data['switches'],
                                   cmap='viridis')
                ax.set_xlabel('Average Delay (ms)')
                ax.set_ylabel('Maximum Delay (ms)')
                ax.set_title(f'{bench_name} - Scheduling Latency')

                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Number of Switches')

                # Add trend line if we have enough points
                if len(data) > 1:
                    try:
                        z = np.polyfit(data['avg_delay_ms'], data['max_delay_ms'], 1)
                        p = np.poly1d(z)
                        ax.plot(data['avg_delay_ms'], p(data['avg_delay_ms']),
                               "r--", alpha=0.8, linewidth=1)
                    except:
                        pass

                # Highlight tasks with highest latency
                top_latency_tasks = data.nlargest(3, 'max_delay_ms')
                for _, task in top_latency_tasks.iterrows():
                    task_name = task['task']
                    if len(task_name) > 20:
                        task_name = task_name[:17] + '...'
                    ax.annotate(task_name,
                                (task['avg_delay_ms'], task['max_delay_ms']),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, alpha=0.7)
            else:
                ax.text(0.5, 0.5, f'No latency data for\n{bench_name}',
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{bench_name} - No Data')

        for i in range(len(self.sched_latency), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / "scheduling_latency.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated scheduling_latency.png")

        self._plot_latency_summary()
        self._plot_task_latency_distribution()


    def _plot_latency_summary(self):
        if not self.sched_latency:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        benchmarks = []
        avg_latencies = []
        max_latencies = []
        total_switches = []
        worst_task_latencies = []
        worst_tasks = []

        for bench_name, data in self.sched_latency.items():
            if not data.empty:
                benchmarks.append(bench_name)
                avg_latencies.append(data['avg_delay_ms'].mean())
                max_latencies.append(data['max_delay_ms'].max())
                total_switches.append(data['switches'].sum())

                worst_task_idx = data['max_delay_ms'].idxmax()
                worst_task = data.loc[worst_task_idx]
                worst_task_latencies.append(worst_task['max_delay_ms'])
                worst_tasks.append(worst_task['task'])

        if benchmarks:
            bars1 = ax1.bar(benchmarks, avg_latencies, alpha=0.7)
            ax1.set_xlabel('Benchmark')
            ax1.set_ylabel('Average Scheduling Latency (ms)')
            ax1.set_title('Average Scheduling Latency Comparison')
            ax1.tick_params(axis='x', rotation=45)

            for bar, val in zip(bars1, avg_latencies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}ms', ha='center', va='bottom', fontsize=8)

            bars2 = ax2.bar(benchmarks, worst_task_latencies, alpha=0.7, color='red')
            ax2.set_xlabel('Benchmark')
            ax2.set_ylabel('Worst Task Latency (ms)')
            ax2.set_title('Worst Task Scheduling Latency')
            ax2.tick_params(axis='x', rotation=45)

            for i, (bar, val, task) in enumerate(zip(bars2, worst_task_latencies, worst_tasks)):
                height = bar.get_height()
                task_short = task[:10] + '...' if len(task) > 10 else task
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}ms\n{task_short}', ha='center', va='bottom', fontsize=8)

            bars3 = ax3.bar(benchmarks, total_switches, alpha=0.7, color='green')
            ax3.set_xlabel('Benchmark')
            ax3.set_ylabel('Total Context Switches')
            ax3.set_title('Total Context Switches')
            ax3.tick_params(axis='x', rotation=45)

            ax4.scatter(total_switches, worst_task_latencies, alpha=0.7, s=60)
            ax4.set_xlabel('Total Context Switches')
            ax4.set_ylabel('Worst Task Latency (ms)')
            ax4.set_title('Latency vs Context Switches')

            for i, bench in enumerate(benchmarks):
                ax4.annotate(bench[:10],
                            (total_switches[i], worst_task_latencies[i]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.7)

        plt.tight_layout()
        plt.savefig(self.output_dir / "latency_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated latency_summary.png")


    def _plot_task_latency_distribution(self):
        if not self.sched_latency:
            return

        latency_data = []
        for bench_name, data in self.sched_latency.items():
            if not data.empty:
                top_tasks = data.nlargest(5, 'max_delay_ms')
                for _, row in top_tasks.iterrows():
                    latency_data.append({
                        'benchmark': bench_name,
                        'task': row['task'],
                        'avg_delay': row['avg_delay_ms'],
                        'max_delay': row['max_delay_ms'],
                        'switches': row['switches']
                    })

        if not latency_data:
            return

        latency_df = pd.DataFrame(latency_data)

        top_latency_tasks = latency_df.nlargest(10, 'max_delay')

        fig, ax = plt.subplots(figsize=(12, 8))

        task_labels = [f"{row['task'][:15]}...\n({row['benchmark']})" if len(row['task']) > 15
                      else f"{row['task']}\n({row['benchmark']})"
                      for _, row in top_latency_tasks.iterrows()]

        x = np.arange(len(task_labels))
        width = 0.35

        avg_delays = top_latency_tasks['avg_delay'].values
        max_delays = top_latency_tasks['max_delay'].values

        ax.bar(x - width/2, avg_delays, width, label='Avg Delay (ms)', alpha=0.7)
        ax.bar(x + width/2, max_delays, width, label='Max Delay (ms)', alpha=0.7, color='red')

        for i, switches in enumerate(top_latency_tasks['switches']):
            ax.text(i, 0.1, f"{switches} sw", ha='center', va='bottom',
                   fontsize=8, rotation=90, alpha=0.7)

        ax.set_ylabel('Delay (ms)')
        ax.set_title('Top Tasks by Scheduling Latency')
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "task_latency_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated task_latency_distribution.png")


    def plot_correlation_analysis(self):
        correlation_data = []

        for bench_name in set(list(self.hyperfine_data.keys()) +
                             list(self.perf_counters.keys()) +
                             list(self.context_stats.keys())):
            row = {'benchmark': bench_name}

            if bench_name in self.hyperfine_data:
                data = self.hyperfine_data[bench_name]
                if isinstance(data, dict) and 'json' in data:
                    results = data['json'].get('results', [])
                    if results:
                        result = results[0]
                        row['execution_time'] = result.get('mean', 0)
                        row['execution_std'] = result.get('stddev', 0)

            if self.resource_data is not None:
                resource_row = self.resource_data[
                    self.resource_data['benchmark'] == bench_name
                ]
                if not resource_row.empty:
                    row['user_time'] = resource_row.iloc[0]['user_time']
                    row['system_time'] = resource_row.iloc[0]['system_time']
                    row['memory_mb'] = resource_row.iloc[0]['max_rss_kb'] / 1024

            if bench_name in self.perf_counters:
                data = self.perf_counters[bench_name]
                row['cycles'] = self._get_perf_metric_value(data, 'cycles')
                row['instructions'] = self._get_perf_metric_value(data, 'instructions')
                row['cache_references'] = self._get_perf_metric_value(data, 'cache-references')
                row['cache_misses'] = self._get_perf_metric_value(data, 'cache-misses')

            if bench_name in self.context_stats:
                data = self.context_stats[bench_name]
                row['context_switches'] = self._get_perf_metric_value(data, 'context-switches')
                row['cpu_migrations'] = self._get_perf_metric_value(data, 'cpu-migrations')

            if bench_name in self.sched_latency:
                sched_data = self.sched_latency[bench_name]
                if not sched_data.empty:
                    row['avg_sched_latency'] = sched_data['avg_delay_ms'].mean()
                    row['max_sched_latency'] = sched_data['max_delay_ms'].max()
                    row['total_switches'] = sched_data['switches'].sum()

            correlation_data.append(row)

        if len(correlation_data) > 1:
            df = pd.DataFrame(correlation_data)

            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 1:
                corr_df = df[numeric_cols].dropna(axis=1, how='all')

                if corr_df.shape[1] > 1:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    correlation_matrix = corr_df.corr()

                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                               square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
                    ax.set_title('Correlation Matrix - Performance Metrics')

                    plt.tight_layout()
                    plt.savefig(self.output_dir / "correlation_analysis.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    print("✓ Generated correlation_analysis.png")
                else:
                    print("Insufficient numeric data for correlation analysis.")
            else:
                print("Insufficient numeric data for correlation analysis.")
        else:
            print("Insufficient benchmarks for correlation analysis.")


    def plot_sched_timeline(self):
        if not self.sched_timeline:
            print("No scheduling timeline data available for plotting.")
            return

        for bench_name, timeline_data in self.sched_timeline.items():
            if timeline_data is None or timeline_data.empty:
                continue

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})

            base_time = timeline_data['timestamp'].min()

            timeline_data['relative_ms'] = (timeline_data['timestamp'] - base_time) * 1000

            task_counts = timeline_data['task_name'].value_counts()
            top_tasks = task_counts.head(10).index.tolist()

            top_task_data = timeline_data[timeline_data['task_name'].isin(top_tasks)]

            colors = plt.cm.tab10(np.linspace(0, 1, len(top_tasks)))
            task_colors = dict(zip(top_tasks, colors))

            for task in top_tasks:
                task_subset = top_task_data[top_task_data['task_name'] == task]

                ax1.scatter(
                    task_subset['relative_ms'],
                    [top_tasks.index(task)] * len(task_subset),
                    s=30,
                    color=task_colors[task],
                    label=task,
                    alpha=0.7
                )

                if 'sched_delay_ms' in task_subset.columns:
                    delays = task_subset['sched_delay_ms'].values
                    if delays.any():
                        sizes = 10 + (delays / delays.max() * 100) if delays.max() > 0 else 10

                        ax1.scatter(
                            task_subset['relative_ms'],
                            [top_tasks.index(task)] * len(task_subset),
                            s=sizes,
                            facecolors='none',
                            edgecolors='red',
                            alpha=0.5
                        )

            if 'sched_delay_ms' in timeline_data.columns:
                delays = timeline_data['sched_delay_ms'].values
                non_zero_delays = delays[delays > 0]

                if len(non_zero_delays) > 0:
                    ax2.hist(non_zero_delays, bins=30, alpha=0.7)
                    ax2.set_xlabel('Scheduling Delay (ms)')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('Scheduling Delay Distribution')
                    ax2.grid(True, alpha=0.3)

                    mean_delay = non_zero_delays.mean()
                    median_delay = np.median(non_zero_delays)
                    ax2.axvline(mean_delay, color='red', linestyle='--', alpha=0.8,
                               label=f'Mean: {mean_delay:.3f}ms')
                    ax2.axvline(median_delay, color='green', linestyle='--', alpha=0.8,
                               label=f'Median: {median_delay:.3f}ms')
                    ax2.legend()
                else:
                    ax2.text(0.5, 0.5, 'No scheduling delay data available',
                            ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'No scheduling delay data available',
                        ha='center', va='center', transform=ax2.transAxes)

            ax1.set_yticks(range(len(top_tasks)))
            ax1.set_yticklabels(top_tasks)
            ax1.set_xlabel('Time from start (ms)')
            ax1.set_title(f'Scheduling Timeline - {bench_name}')
            ax1.grid(True, axis='x', alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / f"{bench_name}_sched_timeline.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Generated {bench_name}_sched_timeline.png")


    def generate_summary_report(self):
        report_path = self.output_dir / "summary_report.txt"

        with open(report_path, 'w') as f:
            f.write("BENCHMARK ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session directory: {self.session_dir}\n\n")

            if self.hyperfine_data:
                f.write("EXECUTION TIMES\n")
                f.write("-" * 20 + "\n")
                for bench_name, data in self.hyperfine_data.items():
                    if isinstance(data, dict) and 'json' in data:
                        results = data['json'].get('results', [])
                        if results:
                            result = results[0]
                            mean_time = result.get('mean', 0)
                            std_time = result.get('stddev', 0)
                            f.write(f"{bench_name}: {mean_time:.4f}s ± {std_time:.4f}s\n")
                f.write("\n")

            if self.resource_data is not None and not self.resource_data.empty:
                f.write("RESOURCE USAGE\n")
                f.write("-" * 20 + "\n")
                for _, row in self.resource_data.iterrows():
                    bench_name = row['benchmark']
                    memory_mb = row['max_rss_kb'] / 1024
                    cpu_time = row['user_time'] + row['system_time']
                    f.write(f"{bench_name}:\n")
                    f.write(f"  Peak Memory: {memory_mb:.1f} MB\n")
                    f.write(f"  CPU Time: {cpu_time:.3f}s (User: {row['user_time']:.3f}s, System: {row['system_time']:.3f}s)\n")
                f.write("\n")

            if self.perf_counters:
                f.write("PERFORMANCE COUNTERS\n")
                f.write("-" * 25 + "\n")
                for bench_name, data in self.perf_counters.items():
                    f.write(f"{bench_name}:\n")

                    cycles = self._get_perf_metric_value(data, 'cycles')
                    instructions = self._get_perf_metric_value(data, 'instructions')
                    cache_refs = self._get_perf_metric_value(data, 'cache-references')
                    cache_misses = self._get_perf_metric_value(data, 'cache-misses')

                    if cycles: f.write(f"  Cycles: {cycles:,.0f}\n")
                    if instructions: f.write(f"  Instructions: {instructions:,.0f}\n")
                    if cache_refs: f.write(f"  Cache References: {cache_refs:,.0f}\n")
                    if cache_misses: f.write(f"  Cache Misses: {cache_misses:,.0f}\n")

                    if instructions and cycles:
                        ipc = instructions / cycles
                        f.write(f"  IPC: {ipc:.3f}\n")

                    if cache_refs and cache_misses:
                        cache_miss_rate = (cache_misses / cache_refs) * 100
                        f.write(f"  Cache Miss Rate: {cache_miss_rate:.2f}%\n")

                    f.write("\n")

            if self.sched_latency:
                f.write("SCHEDULING LATENCY\n")
                f.write("-" * 20 + "\n")
                for bench_name, data in self.sched_latency.items():
                    if not data.empty:
                        avg_latency = data['avg_delay_ms'].mean()
                        max_latency = data['max_delay_ms'].max()
                        total_switches = data['switches'].sum()
                        worst_task_idx = data['max_delay_ms'].idxmax()
                        worst_task = data.loc[worst_task_idx]

                        f.write(f"{bench_name}:\n")
                        f.write(f"  Average Latency: {avg_latency:.4f} ms\n")
                        f.write(f"  Maximum Latency: {max_latency:.4f} ms\n")
                        f.write(f"  Worst Task: {worst_task['task']} ({worst_task['max_delay_ms']:.4f} ms)\n")
                        f.write(f"  Total Context Switches: {total_switches}\n")
                        f.write(f"  Tasks Monitored: {len(data)}\n")
                f.write("\n")

            if self.context_stats:
                f.write("CONTEXT SWITCHES & MIGRATIONS\n")
                f.write("-" * 30 + "\n")
                for bench_name, data in self.context_stats.items():
                    cs_count = self._get_perf_metric_value(data, 'context-switches')
                    mig_count = self._get_perf_metric_value(data, 'cpu-migrations')

                    f.write(f"{bench_name}:")
                    if cs_count is not None:
                        f.write(f" {cs_count:.0f} context switches")
                    if mig_count is not None:
                        f.write(f", {mig_count:.0f} CPU migrations")
                    f.write("\n")

        print(f"✓ Generated summary report: {report_path}")


    def generate_all_plots(self):
        print("Generating benchmark visualization plots...")

        self.plot_execution_times()
        self.plot_resource_usage()
        self.plot_performance_counters()
        self.plot_context_switches()
        self.plot_scheduling_latency()
        self.plot_sched_timeline()
        self.plot_correlation_analysis()
        self.generate_summary_report()

        print("\nAll plots generated successfully!")
        print(f"Output directory: {self.output_dir}")

        generated_files = list(self.output_dir.glob("*.png")) + list(self.output_dir.glob("*.txt"))
        if generated_files:
            print("\nGenerated files:")
            for file in sorted(generated_files):
                print(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize benchmark data collected by the benchmark runner script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_visualizer.py /path/to/session_20250812_100137
  python benchmark_visualizer.py /path/to/session_dir --output /path/to/graphs
  python benchmark_visualizer.py session_20250812_100137 --plots execution,latency
        """
    )

    parser.add_argument(
        'session_dir',
        help='Path to the benchmark session directory'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output directory for graphs (default: session_dir/graphs)',
        default=None
    )

    parser.add_argument(
        '--plots',
        help='Comma-separated list of plot types to generate (execution,resource,perf,context,latency,timeline,correlation,all)',
        default='all'
    )

    parser.add_argument(
        '--format',
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Output format for plots (default: png)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output images (default: 300)'
    )

    args = parser.parse_args()

    session_path = Path(args.session_dir)
    if not session_path.exists():
        print(f"Error: Session directory '{session_path}' does not exist.")
        sys.exit(1)

    if not session_path.is_dir():
        print(f"Error: '{session_path}' is not a directory.")
        sys.exit(1)

    try:
        visualizer = BenchmarkVisualizer(args.session_dir, args.output)

        plt.rcParams['figure.dpi'] = args.dpi
        plt.rcParams['savefig.format'] = args.format

        plot_types = [p.strip().lower() for p in args.plots.split(',')]

        if 'all' in plot_types:
            visualizer.generate_all_plots()
        else:
            if 'execution' in plot_types:
                visualizer.plot_execution_times()
            if 'resource' in plot_types:
                visualizer.plot_resource_usage()
            if 'perf' in plot_types:
                visualizer.plot_performance_counters()
            if 'context' in plot_types:
                visualizer.plot_context_switches()
            if 'latency' in plot_types:
                visualizer.plot_scheduling_latency()
            if 'timeline' in plot_types:
                visualizer.plot_sched_timeline()
            if 'correlation' in plot_types:
                visualizer.plot_correlation_analysis()

            visualizer.generate_summary_report()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        import seaborn as sns
    except ImportError as e:
        print(f"Error: Missing required dependency: {e}")
        print("\nPlease install required packages:")
        print("pip install matplotlib pandas numpy seaborn")
        sys.exit(1)

    main()
