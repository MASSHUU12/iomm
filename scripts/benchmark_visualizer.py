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


    def _load_perf_data(self):
        for bench_dir in self.session_dir.iterdir():
            if bench_dir.is_dir():
                bench_name = bench_dir.name

                perf_file = bench_dir / "perf_counters.csv"
                if perf_file.exists():
                    try:
                        df = pd.read_csv(perf_file, header=None,
                                       names=['value', 'unit', 'event', 'runtime', 'percentage'])
                        self.perf_counters[bench_name] = df
                    except Exception as e:
                        print(f"Warning: Failed to load {perf_file}: {e}")

                context_file = bench_dir / "context_stats.csv"
                if context_file.exists():
                    try:
                        df = pd.read_csv(context_file, header=None,
                                       names=['value', 'unit', 'event', 'runtime', 'percentage'])
                        self.context_stats[bench_name] = df
                    except Exception as e:
                        print(f"Warning: Failed to load {context_file}: {e}")


    def _load_sched_data(self):
        for bench_dir in self.session_dir.iterdir():
            if bench_dir.is_dir():
                bench_name = bench_dir.name
                sched_file = bench_dir / "sched_latency.txt"

                if sched_file.exists():
                    try:
                        latency_data = self._parse_sched_latency(sched_file)
                        if latency_data:
                            self.sched_latency[bench_name] = latency_data
                    except Exception as e:
                        print(f"Warning: Failed to parse {sched_file}: {e}")


    def _parse_sched_latency(self, file_path: Path) -> Optional[pd.DataFrame]:
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            lines = content.split('\n')
            data_started = False
            rows = []

            for line in lines:
                line = line.strip()
                if 'Task' in line and 'Runtime ms' in line:
                    data_started = True
                    continue

                if data_started and line and not line.startswith('-'):
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 5:
                        task_info = parts[0].strip()
                        runtime = self._extract_number(parts[1])
                        switches = self._extract_number(parts[2])
                        avg_delay = self._extract_number(parts[3])
                        max_delay = self._extract_number(parts[4])

                        if all(x is not None for x in [runtime, switches, avg_delay, max_delay]):
                            rows.append({
                                'task': task_info,
                                'runtime_ms': runtime,
                                'switches': switches,
                                'avg_delay_ms': avg_delay,
                                'max_delay_ms': max_delay
                            })

            return pd.DataFrame(rows) if rows else None

        except Exception as e:
            print(f"Error parsing sched latency file {file_path}: {e}")
            return None


    def _extract_number(self, text: str) -> Optional[float]:
        try:
            cleaned = re.sub(r'[^\d\.-]', '', text.strip())
            return float(cleaned) if cleaned else None
        except:
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
                  'branch-instructions', 'branch-misses']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break

            benchmark_names = []
            values = []

            for bench_name, data in self.perf_counters.items():
                metric_row = data[data['event'].str.contains(metric, na=False)]
                if not metric_row.empty:
                    try:
                        value = float(metric_row.iloc[0]['value'].replace(',', ''))
                        benchmark_names.append(bench_name)
                        values.append(value)
                    except (ValueError, AttributeError):
                        continue

            if benchmark_names and values:
                bars = axes[i].bar(benchmark_names, values, alpha=0.7)
                axes[i].set_xlabel('Benchmark')
                axes[i].set_ylabel('Count')
                axes[i].set_title(f'{metric.replace("-", " ").title()}')
                axes[i].tick_params(axis='x', rotation=45)

                if len(values) <= 5:
                    for bar, val in zip(bars, values):
                        height = bar.get_height()
                        axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{val:.0f}', ha='center', va='bottom', fontsize=8)
            else:
                axes[i].text(0.5, 0.5, f'No {metric} data',
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f'{metric.replace("-", " ").title()} (No Data)')

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

            cs_row = data[data['event'].str.contains('context-switches', na=False)]
            context_switches.append(
                float(cs_row.iloc[0]['value'].replace(',', '')) if not cs_row.empty else 0
            )

            mig_row = data[data['event'].str.contains('cpu-migrations', na=False)]
            cpu_migrations.append(
                float(mig_row.iloc[0]['value'].replace(',', '')) if not mig_row.empty else 0
            )

            pf_row = data[data['event'].str.contains('page-faults', na=False)]
            page_faults.append(
                float(pf_row.iloc[0]['value'].replace(',', '')) if not pf_row.empty else 0
            )

            tc_row = data[data['event'].str.contains('task-clock', na=False)]
            if not tc_row.empty:
                tc_val = tc_row.iloc[0]['value'].replace(',', '')
                task_clock.append(float(tc_val) if tc_val else 0)
            else:
                task_clock.append(0)

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
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        for i, (bench_name, data) in enumerate(self.sched_latency.items()):
            if i >= len(axes):
                break

            ax = axes[i]

            if not data.empty and 'avg_delay_ms' in data.columns and 'max_delay_ms' in data.columns:
                scatter = ax.scatter(data['avg_delay_ms'], data['max_delay_ms'],
                                   s=data['switches']/10, alpha=0.6)
                ax.set_xlabel('Average Delay (ms)')
                ax.set_ylabel('Maximum Delay (ms)')
                ax.set_title(f'{bench_name} - Scheduling Latency')

                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Number of Switches')

                if len(data) > 1:
                    z = np.polyfit(data['avg_delay_ms'], data['max_delay_ms'], 1)
                    p = np.poly1d(z)
                    ax.plot(data['avg_delay_ms'], p(data['avg_delay_ms']),
                           "r--", alpha=0.8, linewidth=1)
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


    def _plot_latency_summary(self):
        if not self.sched_latency:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        benchmarks = []
        avg_latencies = []
        max_latencies = []
        total_switches = []

        for bench_name, data in self.sched_latency.items():
            if not data.empty:
                benchmarks.append(bench_name)
                avg_latencies.append(data['avg_delay_ms'].mean())
                max_latencies.append(data['max_delay_ms'].max())
                total_switches.append(data['switches'].sum())

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

            bars2 = ax2.bar(benchmarks, max_latencies, alpha=0.7, color='red')
            ax2.set_xlabel('Benchmark')
            ax2.set_ylabel('Maximum Scheduling Latency (ms)')
            ax2.set_title('Maximum Scheduling Latency Comparison')
            ax2.tick_params(axis='x', rotation=45)

            for bar, val in zip(bars2, max_latencies):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}ms', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / "latency_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated latency_summary.png")


    def plot_correlation_analysis(self):
        correlation_data = []

        for bench_name in self.hyperfine_data.keys():
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

            if bench_name in self.context_stats:
                data = self.context_stats[bench_name]
                cs_row = data[data['event'].str.contains('context-switches', na=False)]
                if not cs_row.empty:
                    row['context_switches'] = float(cs_row.iloc[0]['value'].replace(',', ''))

                mig_row = data[data['event'].str.contains('cpu-migrations', na=False)]
                if not mig_row.empty:
                    row['cpu_migrations'] = float(mig_row.iloc[0]['value'].replace(',', ''))

            if bench_name in self.sched_latency:
                sched_data = self.sched_latency[bench_name]
                if not sched_data.empty:
                    row['avg_sched_latency'] = sched_data['avg_delay_ms'].mean()
                    row['max_sched_latency'] = sched_data['max_delay_ms'].max()

            correlation_data.append(row)

        if len(correlation_data) > 1:
            df = pd.DataFrame(correlation_data)

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = numeric_cols.drop('benchmark', errors='ignore')

            if len(numeric_cols) > 1:
                corr_df = df[numeric_cols]

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
            print("Insufficient benchmarks for correlation analysis.")


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

            if self.sched_latency:
                f.write("SCHEDULING LATENCY\n")
                f.write("-" * 20 + "\n")
                for bench_name, data in self.sched_latency.items():
                    if not data.empty:
                        avg_latency = data['avg_delay_ms'].mean()
                        max_latency = data['max_delay_ms'].max()
                        total_switches = data['switches'].sum()
                        f.write(f"{bench_name}:\n")
                        f.write(f"  Average Latency: {avg_latency:.4f} ms\n")
                        f.write(f"  Maximum Latency: {max_latency:.4f} ms\n")
                        f.write(f"  Total Context Switches: {total_switches}\n")
                f.write("\n")

            if self.context_stats:
                f.write("CONTEXT SWITCHES & MIGRATIONS\n")
                f.write("-" * 30 + "\n")
                for bench_name, data in self.context_stats.items():
                    cs_row = data[data['event'].str.contains('context-switches', na=False)]
                    mig_row = data[data['event'].str.contains('cpu-migrations', na=False)]

                    if not cs_row.empty:
                        cs_count = cs_row.iloc[0]['value']
                        f.write(f"{bench_name}: {cs_count} context switches")

                        if not mig_row.empty:
                            mig_count = mig_row.iloc[0]['value']
                            f.write(f", {mig_count} CPU migrations")
                        f.write("\n")

        print(f"✓ Generated summary report: {report_path}")


    def generate_all_plots(self):
        print("Generating benchmark visualization plots...")

        self.plot_execution_times()
        self.plot_resource_usage()
        self.plot_performance_counters()
        self.plot_context_switches()
        self.plot_scheduling_latency()
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
        help='Comma-separated list of plot types to generate (execution,resource,perf,context,latency,correlation,all)',
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
            if 'correlation' in plot_types:
                visualizer.plot_correlation_analysis()

            visualizer.generate_summary_report()

    except Exception as e:
        print(f"Error: {e}")
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
