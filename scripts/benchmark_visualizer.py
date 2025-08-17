#!/usr/bin/env python3

import argparse
import json
import re
import sys
import gc
import time
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any
import warnings
from functools import wraps

warnings.filterwarnings('ignore')

def _with_memory_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        result = func(*args, **kwargs)
        gc.collect()
        return result
    return wrapper

class BenchmarkVisualizer:
    def __init__(self, session_dir: str, output_dir: Optional[str] = None):
        self.session_dir = Path(session_dir)
        self.output_dir = Path(output_dir) if output_dir else self.session_dir / "graphs"
        self.output_dir.mkdir(exist_ok=True)

        self.hyperfine_paths = {}
        self.resource_path = None
        self.perf_counter_paths = {}
        self.context_stats_paths = {}
        self.sched_latency_paths = {}
        self.sched_map_paths = {}
        self.sched_timeline_paths = {}

        self.debug_mode = False
        self._scan_data_files()


    def _scan_data_files(self):
        print(f"Scanning data files in: {self.session_dir}")

        resource_file = self.session_dir / "resource_usage.csv"
        if resource_file.exists():
            self.resource_path = resource_file

        for bench_dir in self.session_dir.iterdir():
            if bench_dir.is_dir():
                bench_name = bench_dir.name

                csv_file = bench_dir / "hyperfine_time.csv"
                json_file = bench_dir / "hyperfine_time.json"

                if csv_file.exists() or json_file.exists():
                    self.hyperfine_paths[bench_name] = {}
                    if csv_file.exists():
                        self.hyperfine_paths[bench_name]['csv'] = csv_file
                    if json_file.exists():
                        self.hyperfine_paths[bench_name]['json'] = json_file

                perf_file = bench_dir / "perf_counters.csv"
                if perf_file.exists():
                    self.perf_counter_paths[bench_name] = perf_file

                context_file = bench_dir / "context_stats.csv"
                if context_file.exists():
                    self.context_stats_paths[bench_name] = context_file

                sched_latency_file = bench_dir / "sched_latency.txt"
                if sched_latency_file.exists():
                    self.sched_latency_paths[bench_name] = sched_latency_file

                sched_map_file = bench_dir / "sched_map.txt"
                if sched_map_file.exists():
                    self.sched_map_paths[bench_name] = sched_map_file

                sched_timeline_file = bench_dir / "sched_timeline.txt"
                if sched_timeline_file.exists():
                    self.sched_timeline_paths[bench_name] = sched_timeline_file

        for file_name, path_dict in [
            ("sched_latency.txt", self.sched_latency_paths),
            ("sched_map.txt", self.sched_map_paths),
            ("sched_timeline.txt", self.sched_timeline_paths)
        ]:
            file_path = self.session_dir / file_name
            if file_path.exists():
                path_dict["session_root"] = file_path

        print(f"File scanning complete. Graphs will be saved to: {self.output_dir}")


    def _load_hyperfine_data_for_bench(self, bench_name: str) -> dict:
        result = {}

        if bench_name not in self.hyperfine_paths:
            return result

        paths = self.hyperfine_paths[bench_name]

        if 'csv' in paths:
            try:
                import pandas as pd
                result['df'] = pd.read_csv(paths['csv'])
            except Exception as e:
                print(f"Warning: Failed to load {paths['csv']}: {e}")

        if 'json' in paths:
            try:
                with open(paths['json'], 'r') as f:
                    result['json'] = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {paths['json']}: {e}")

        return result


    def _load_resource_data(self):
        if self.resource_path is None:
            return None

        try:
            import pandas as pd
            resource_data = pd.read_csv(self.resource_path)

            if 'timestamp' in resource_data.columns:
                resource_data['datetime'] = pd.to_datetime(
                    resource_data['timestamp'], unit='s'
                )

            if 'benchmark' in resource_data.columns:
                resource_data['benchmark'] = resource_data['benchmark'].astype('category')

            return resource_data
        except Exception as e:
            print(f"Warning: Failed to load {self.resource_path}: {e}")
            return None


    def _parse_perf_csv(self, file_path: Path) -> Optional[list]:
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
            return rows
        except Exception as e:
            print(f"Error parsing perf CSV {file_path}: {e}")
            return None


    def _get_perf_data(self, bench_name: str):
        if bench_name not in self.perf_counter_paths:
            return None

        rows = self._parse_perf_csv(self.perf_counter_paths[bench_name])
        if not rows:
            return None

        import pandas as pd
        return pd.DataFrame(rows)


    def _get_context_stats(self, bench_name: str):
        if bench_name not in self.context_stats_paths:
            return None

        rows = self._parse_perf_csv(self.context_stats_paths[bench_name])
        if not rows:
            return None

        import pandas as pd
        return pd.DataFrame(rows)


    def _parse_sched_latency(self, file_path: Path) -> Optional[List[dict]]:
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

            return rows
        except Exception as e:
            print(f"Error parsing sched latency file {file_path}: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
            return None


    def _get_sched_latency(self, bench_name: str):
        if bench_name not in self.sched_latency_paths:
            return None

        rows = self._parse_sched_latency(self.sched_latency_paths[bench_name])
        if not rows:
            return None

        import pandas as pd
        return pd.DataFrame(rows)


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


    def _get_sched_map(self, bench_name: str):
        if bench_name not in self.sched_map_paths:
            return None

        return self._parse_sched_map(self.sched_map_paths[bench_name])


    def _parse_sched_timeline(self, file_path: Path) -> Optional[List[dict]]:
        try:
            rows = []

            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip().startswith("-------"):
                        break

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        parts = line.split()

                        if len(parts) < 6:
                            if self.debug_mode:
                                print(f"DEBUG: Skipping line with insufficient parts: {line}")
                            continue

                        timestamp_str = parts[0]
                        timestamp = float(timestamp_str)

                        cpu_str = parts[1]
                        cpu = -1
                        cpu_match = re.search(r'\[(\d+)\]', cpu_str)
                        if cpu_match:
                            cpu = int(cpu_match.group(1))

                        task_match = re.search(r'(\S+(?:\s+\S+)*?)\[(\d+)\]', line)
                        if task_match:
                            full_task_part = task_match.group(0)
                            task_name = task_match.group(1).strip()
                            pid = int(task_match.group(2))

                            match_end = line.find(full_task_part) + len(full_task_part)
                            remaining_line = line[match_end:].strip()
                            timing_parts = remaining_line.split()

                            if len(timing_parts) >= 3:
                                wait_time_ms = float(timing_parts[0])
                                sch_delay_ms = float(timing_parts[1])
                                run_time_ms = float(timing_parts[2])
                            else:
                                wait_time_ms = float(parts[-3])
                                sch_delay_ms = float(parts[-2])
                                run_time_ms = float(parts[-1])
                        else:
                            if len(parts) >= 6:
                                task_name = parts[2]
                                pid = -1
                                pid_match = re.search(r'\[(\d+)\]', task_name)
                                if pid_match:
                                    pid = int(pid_match.group(1))
                                    task_name = re.sub(r'\[\d+\]', '', task_name).strip()

                                wait_time_ms = float(parts[-3])
                                sch_delay_ms = float(parts[-2])
                                run_time_ms = float(parts[-1])
                            else:
                                if self.debug_mode:
                                    print(f"DEBUG: Could not parse task info from line: {line}")
                                continue

                        rows.append({
                            'timestamp': timestamp,
                            'cpu': cpu,
                            'task_name': task_name,
                            'pid': pid,
                            'wait_time_ms': wait_time_ms,
                            'sched_delay_ms': sch_delay_ms,
                            'run_time_ms': run_time_ms
                        })

                    except Exception as e:
                        if self.debug_mode:
                            print(f"DEBUG: Error parsing timeline line: {line}")
                            print(f"DEBUG: Exception: {e}")
                        continue

            if not rows:
                print(f"No valid timeline data rows found in: {file_path}")
                return None

            return rows
        except Exception as e:
            print(f"Error parsing sched timeline file {file_path}: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
            return None


    def _get_sched_timeline(self, bench_name: str, sample_rate=1.0):
        if bench_name not in self.sched_timeline_paths:
            return None

        rows = self._parse_sched_timeline(self.sched_timeline_paths[bench_name])
        if not rows:
            return None

        import pandas as pd
        df = pd.DataFrame(rows)

        if sample_rate < 1.0 and len(df) > 1000:
            return self._smart_sample(df, max_points=int(len(df) * sample_rate))

        return df


    def _smart_sample(self, data, max_points=5000):
        import pandas as pd

        if len(data) <= max_points:
            return data

        important = data[data['sched_delay_ms'] > data['sched_delay_ms'].quantile(0.95)]
        to_sample = len(data) - len(important)
        sample_rate = (max_points - len(important)) / to_sample if to_sample > 0 else 0

        if sample_rate <= 0:
            return important.sample(n=max_points)

        sampled = data.drop(important.index).sample(frac=sample_rate)
        return pd.concat([important, sampled])


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


    def _get_perf_metric_value(self, data, metric_name: str) -> Optional[float]:
        if isinstance(data, list):  # If data is a list of dicts
            for row in data:
                if metric_name.lower() in row.get('event', '').lower():
                    try:
                        value_str = row.get('value', '0')
                        cleaned_value = value_str.replace(',', '')
                        return float(cleaned_value) if cleaned_value else None
                    except (ValueError, TypeError):
                        pass
            return None
        else:  # If data is a dataframe
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
        if not self.hyperfine_paths:
            print("No hyperfine timing data available for plotting.")
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        benchmarks = []
        means = []
        stds = []
        medians = []
        mins = []
        maxs = []
        distributions = []
        labels = []

        for bench_name in self.hyperfine_paths:
            data = self._load_hyperfine_data_for_bench(bench_name)

            if 'json' in data:
                results = data['json'].get('results', [])
                if results:
                    result = results[0]  # First (and usually only) result
                    benchmarks.append(bench_name)
                    means.append(result.get('mean', 0))
                    stds.append(result.get('stddev', 0))
                    medians.append(result.get('median', 0))
                    mins.append(result.get('min', 0))
                    maxs.append(result.get('max', 0))

                    times = result.get('times', [])
                    if times:
                        distributions.append(times)
                        labels.append(bench_name)
            elif 'df' in data:
                df = data['df']
                if not df.empty and 'mean' in df.columns:
                    benchmarks.append(bench_name)
                    means.append(df['mean'].iloc[0])
                    stds.append(df['stddev'].iloc[0] if 'stddev' in df.columns else 0)
                    medians.append(df['median'].iloc[0] if 'median' in df.columns else df['mean'].iloc[0])
                    mins.append(df['min'].iloc[0] if 'min' in df.columns else df['mean'].iloc[0])
                    maxs.append(df['max'].iloc[0] if 'max' in df.columns else df['mean'].iloc[0])

                    distributions.append([df['mean'].iloc[0]] * 10)  # Placeholder distribution
                    labels.append(bench_name)

            del data

        if not benchmarks:
            print("No timing data found for execution time plots.")
            return

        clean_names, name_mapping = self._get_clean_benchmark_names(benchmarks)

        use_log_scale = False
        if means and max(means) > 0 and min(means) > 0:
            ratio = max(means) / min(means)
            use_log_scale = ratio > 10

        fig = plt.figure(figsize=(18, 15))

        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(clean_names, means, yerr=stds, capsize=5, alpha=0.8, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('Benchmark', fontweight='bold')
        ax1.set_ylabel('Execution Time (seconds)', fontweight='bold')
        ax1.set_title(f'Mean Execution Times with Standard Deviation\n'
                     f'(fastest: {min(means):.3f}s, slowest: {max(means):.3f}s)',
                     pad=15)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        if use_log_scale:
            ax1.set_yscale('log')
            ax1.set_ylabel('Execution Time (seconds) - Log Scale', fontweight='bold')

        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            offset = std if not use_log_scale else height * 0.1
            ax1.annotate(f'{mean:.3f}s ± {std:.3f}s',
                        xy=(bar.get_x() + bar.get_width()/2, height + offset),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax2 = fig.add_subplot(gs[0, 1])
        if distributions and any(len(d) > 1 for d in distributions):
            valid_distributions = []
            valid_labels = []
            for dist, label in zip(distributions, labels):
                if len(dist) > 1:
                    valid_distributions.append(dist)
                    clean_label = self._clean_benchmark_name(label)
                    valid_labels.append(clean_label)

            if valid_distributions:
                bp = ax2.boxplot(valid_distributions, labels=valid_labels, patch_artist=True)
                for i, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(plt.cm.tab10(i % 10))
                    patch.set_alpha(0.7)
                for flier in bp['fliers']:
                    flier.set(marker='o', markerfacecolor='red', markersize=8)

                ax2.set_xlabel('Benchmark', fontweight='bold')
                ax2.set_ylabel('Execution Time (seconds)', fontweight='bold')
                ax2.set_title('Execution Time Distributions (Box Plot)')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)

                if use_log_scale:
                    ax2.set_yscale('log')
                    ax2.set_ylabel('Execution Time (seconds) - Log Scale', fontweight='bold')

                for i, dist in enumerate(valid_distributions):
                    if len(dist) > 3:
                        p95 = np.percentile(dist, 95)
                        ax2.annotate(f'95%: {p95:.2f}s',
                                    xy=(i+1, p95),
                                    xytext=(0, 5),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)
            else:
                ax2.text(0.5, 0.5, 'Insufficient data for\ndistribution analysis',
                        transform=ax2.transAxes, ha='center', va='center', fontsize=12)
                ax2.set_title('Execution Time Distributions (Insufficient Data)')
        else:
            ax2.text(0.5, 0.5, 'No distribution data available',
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title('Execution Time Distributions (No Data)')

        ax3 = fig.add_subplot(gs[1, 0])
        if mins and maxs:
            x_pos = np.arange(len(clean_names))
            error_bars = ax3.errorbar(x_pos, means, yerr=[np.array(means) - np.array(mins),
                                                np.array(maxs) - np.array(means)],
                                    fmt='o', capsize=5, capthick=2, markersize=8, alpha=0.8,
                                    ecolor='red', markerfacecolor='blue', markeredgecolor='black')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(clean_names, rotation=45)
            ax3.set_xlabel('Benchmark', fontweight='bold')
            ax3.set_ylabel('Execution Time (seconds)', fontweight='bold')
            ax3.set_title('Min/Max Range with Mean')
            ax3.grid(True, alpha=0.3)

            if use_log_scale:
                ax3.set_yscale('log')
                ax3.set_ylabel('Execution Time (seconds) - Log Scale', fontweight='bold')

            for i, (clean_name, mean_val, min_val, max_val) in enumerate(zip(clean_names, means, mins, maxs)):
                range_val = max_val - min_val
                offset = mean_val * 0.1 if use_log_scale else 0
                ax3.annotate(f'Range: {range_val:.4f}s',
                            xy=(i, mean_val + offset),
                            xytext=(5, 10),
                            textcoords='offset points', fontsize=9, alpha=0.8,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
        else:
            ax3.text(0.5, 0.5, 'No min/max data available',
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('Min/Max Range (No Data)')

        ax4 = fig.add_subplot(gs[1, 1])
        if means:
            fastest_time = min(means)
            relative_performance = [mean / fastest_time for mean in means]

            colors = ['green' if rp == 1.0 else 'orange' if rp < 1.5 else 'red' for rp in relative_performance]
            bars = ax4.bar(clean_names, relative_performance, color=colors, alpha=0.8)

            ax4.set_xlabel('Benchmark', fontweight='bold')
            ax4.set_ylabel('Relative Performance (vs fastest)', fontweight='bold')
            ax4.set_title('Relative Performance Comparison')
            ax4.tick_params(axis='x', rotation=45)
            ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Baseline (fastest)')
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='upper left', bbox_to_anchor=(0, 1.15), frameon=True, fancybox=True)

            for bar, rp, mean in zip(bars, relative_performance, means):
                height = bar.get_height()
                if rp == 1.0:
                    label = f'Fastest\n{mean:.3f}s'
                else:
                    slower_pct = (rp - 1.0) * 100
                    label = f'+{slower_pct:.1f}%\n{mean:.3f}s'
                ax4.annotate(label,
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

        ax_violin = fig.add_subplot(gs[2, :])
        if distributions and any(len(d) > 5 for d in distributions):
            valid_distributions = []
            valid_labels = []
            for dist, label in zip(distributions, labels):
                if len(dist) > 5:
                    valid_distributions.append(dist)
                    clean_label = self._clean_benchmark_name(label)
                    valid_labels.append(clean_label)

            if valid_distributions:
                violin_parts = ax_violin.violinplot(valid_distributions, showmedians=True,
                                                  vert=True, widths=0.8, showextrema=True)

                for i, pc in enumerate(violin_parts['bodies']):
                    pc.set_facecolor(plt.cm.tab10(i % 10))
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')

                violin_parts['cmedians'].set_color('black')
                violin_parts['cmedians'].set_linewidth(2)

                ax_violin.set_xticks(np.arange(1, len(valid_labels)+1))
                ax_violin.set_xticklabels(valid_labels, rotation=45)

                ax_violin.set_xlabel('Benchmark', fontweight='bold')
                ax_violin.set_ylabel('Execution Time (seconds)', fontweight='bold')
                ax_violin.set_title('Execution Time Distribution (Violin Plot)')
                ax_violin.grid(True, axis='y', alpha=0.3)

                if use_log_scale:
                    ax_violin.set_yscale('log')
                    ax_violin.set_ylabel('Execution Time (seconds) - Log Scale', fontweight='bold')
            else:
                ax_violin.text(0.5, 0.5, 'Insufficient data for violin plot\n(need more samples per benchmark)',
                              transform=ax_violin.transAxes, ha='center', va='center', fontsize=12)
                ax_violin.set_title('Execution Time Distribution - Violin Plot (Insufficient Data)')
        else:
            ax_violin.text(0.5, 0.5, 'No distribution data available for violin plot',
                          transform=ax_violin.transAxes, ha='center', va='center', fontsize=12)
            ax_violin.set_title('Execution Time Distribution - Violin Plot (No Data)')

        plt.tight_layout()
        plt.savefig(self.output_dir / "execution_times.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Generated execution_times.png")

        self._generate_execution_summary_table(benchmarks, clean_names, means, stds, medians, mins, maxs)

        gc.collect()


    def plot_performance_trends(self):
        resource_data = self._load_resource_data()
        if resource_data is None or resource_data.empty:
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy import stats
        import seaborn as sns

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        if 'datetime' in resource_data.columns:
            resource_data_sorted = resource_data.sort_values('datetime').reset_index(drop=True)

            x = np.arange(len(resource_data_sorted))
            y = resource_data_sorted['elapsed_time']

            if len(x) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                trend_line = slope * x + intercept

                from scipy import stats
                n = len(x)
                y_hat = slope * x + intercept
                tinv = stats.t.ppf(1-0.05/2, n-2)  # 95% confidence
                x_mean = np.mean(x)
                s_err = np.sqrt(np.sum((y - y_hat)**2) / (n-2))
                conf = tinv * s_err * np.sqrt(1/n + (x-x_mean)**2 / np.sum((x-x_mean)**2))

                ax1.scatter(x, y, alpha=0.8, s=60, c='blue', edgecolors='navy')
                ax1.plot(x, trend_line, 'r-', linewidth=2, alpha=0.8,
                        label=f'Trend: slope={slope:.4f}, R²={r_value**2:.3f}')
                ax1.fill_between(x, y_hat-conf, y_hat+conf, alpha=0.2, color='r', label='95% Confidence')
                ax1.set_xlabel('Benchmark Sequence', fontweight='bold')
                ax1.set_ylabel('Execution Time (seconds)', fontweight='bold')
                ax1.set_title('Performance Trend Analysis', fontweight='bold', fontsize=14)

                sig_text = f"p-value: {p_value:.4f}"
                if p_value < 0.05:
                    sig_text += " (significant trend)"
                else:
                    sig_text += " (no significant trend)"
                ax1.text(0.05, 0.95, sig_text, transform=ax1.transAxes,
                         fontsize=10, va='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax1.legend(loc='best', frameon=True, fancybox=True)
                ax1.grid(True, alpha=0.3)

            if len(resource_data_sorted) > 1:
                window_size = min(3, len(y))
                rolling_mean = pd.Series(y).rolling(window=window_size).mean()
                rolling_std = pd.Series(y).rolling(window=window_size).std()

                ax2.plot(x, y, 'bo-', alpha=0.7, label='Actual')
                ax2.plot(x, rolling_mean, 'r-', linewidth=2, label=f'{window_size}-point Rolling Mean')
                ax2.fill_between(x,
                              rolling_mean - rolling_std,
                              rolling_mean + rolling_std,
                              alpha=0.3, label='±1 Std Dev')
                ax2.set_xlabel('Benchmark Sequence', fontweight='bold')
                ax2.set_ylabel('Execution Time (seconds)', fontweight='bold')
                ax2.set_title('Performance Stability Analysis', fontweight='bold', fontsize=14)
                ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
                ax2.grid(True, alpha=0.3)

            cpu_time = resource_data_sorted['user_time'] + resource_data_sorted['system_time']
            memory_mb = resource_data_sorted['max_rss_kb'] / 1024

            scatter = ax3.scatter(cpu_time, memory_mb,
                                c=resource_data_sorted['elapsed_time'],
                                s=80, alpha=0.8, cmap='viridis',
                                edgecolor='black', linewidth=0.5)
            ax3.set_xlabel('CPU Time (seconds)', fontweight='bold')
            ax3.set_ylabel('Peak Memory (MB)', fontweight='bold')
            ax3.set_title('Resource Usage Pattern', fontweight='bold', fontsize=14)
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Execution Time (s)', fontweight='bold')

            if len(cpu_time) <= 10:  # Only annotate if not too crowded
                for i, (x_val, y_val) in enumerate(zip(cpu_time, memory_mb)):
                    ax3.annotate(f"#{i+1}",
                                xy=(x_val, y_val),
                                xytext=(5, 5),
                                textcoords='offset points',
                                fontsize=8)

            ax3.grid(True, alpha=0.3)

            sns.histplot(resource_data_sorted['elapsed_time'], bins=min(15, len(resource_data_sorted)),
                      kde=True, ax=ax4, color='skyblue', edgecolor='black', alpha=0.7, line_kws={'linewidth': 2})

            percentiles = [50, 90, 95]
            colors = ['green', 'orange', 'red']
            for p, color in zip(percentiles, colors):
                percentile_val = np.percentile(resource_data_sorted['elapsed_time'], p)
                ax4.axvline(percentile_val, color=color, linestyle='--',
                           alpha=0.8, label=f'{p}th percentile: {percentile_val:.2f}s')

            ax4.axvline(resource_data_sorted['elapsed_time'].mean(),
                    color='blue', linestyle='-', linewidth=2, label=f'Mean: {resource_data_sorted["elapsed_time"].mean():.2f}s')
            ax4.axvline(resource_data_sorted['elapsed_time'].median(),
                    color='purple', linestyle='-.', linewidth=2, label=f'Median: {resource_data_sorted["elapsed_time"].median():.2f}s')

            ax4.set_xlabel('Execution Time (seconds)', fontweight='bold')
            ax4.set_ylabel('Frequency', fontweight='bold')
            ax4.set_title('Performance Distribution', fontweight='bold', fontsize=14)
            ax4.legend(loc='best', frameon=True, fancybox=True)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_trends.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Generated performance_trends.png")


    def _generate_execution_summary_table(self, benchmarks, clean_names, means, stds, medians, mins, maxs):
        import pandas as pd

        if not benchmarks:
            return

        summary_data = []
        for i, (original_name, clean_name) in enumerate(zip(benchmarks, clean_names)):
            row = {
                'Original Name': original_name,
                'Display Name': clean_name,
                'Mean (s)': f"{means[i]:.4f}",
                'Std Dev (s)': f"{stds[i]:.4f}" if i < len(stds) else "N/A",
                'Median (s)': f"{medians[i]:.4f}" if i < len(medians) else "N/A",
                'Min (s)': f"{mins[i]:.4f}" if i < len(mins) else "N/A",
                'Max (s)': f"{maxs[i]:.4f}" if i < len(maxs) else "N/A",
                'CV (%)': f"{(stds[i]/means[i]*100):.2f}" if i < len(stds) and means[i] > 0 else "N/A"
            }
            summary_data.append(row)

        df = pd.DataFrame(summary_data)

        csv_path = self.output_dir / "execution_times_summary.csv"
        df.to_csv(csv_path, index=False)

        txt_path = self.output_dir / "execution_times_summary.txt"
        with open(txt_path, 'w') as f:
            f.write("EXECUTION TIMES SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(df.to_string(index=False))
            fastest_idx = means.index(min(means))
            slowest_idx = means.index(max(means))
            f.write(f"\n\nFastest benchmark: {clean_names[fastest_idx]} ({means[fastest_idx]:.4f}s)")
            f.write(f"\nSlowest benchmark: {clean_names[slowest_idx]} ({means[slowest_idx]:.4f}s)")
            if len(means) > 1:
                speed_ratio = max(means) / min(means)
                f.write(f"\nSpeed ratio (slowest/fastest): {speed_ratio:.2f}x")

        print("✓ Generated execution_times_summary.csv and execution_times_summary.txt")


    def plot_resource_usage(self):
        resource_data = self._load_resource_data()
        if resource_data is None or resource_data.empty:
            print("No resource usage data available for plotting.")
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        original_benchmarks = resource_data['benchmark'].values
        clean_benchmarks, name_mapping = self._get_clean_benchmark_names(original_benchmarks)

        user_time = resource_data['user_time'].values
        system_time = resource_data['system_time'].values

        x = np.arange(len(clean_benchmarks))
        width = 0.35

        bars1 = ax1.bar(x - width/2, user_time, width, label='User Time', alpha=0.8, color='#3274A1')
        bars2 = ax1.bar(x + width/2, system_time, width, label='System Time', alpha=0.8, color='#E1812C')
        ax1.set_xlabel('Benchmark', fontweight='bold')
        ax1.set_ylabel('Time (seconds)', fontweight='bold')
        ax1.set_title('CPU Time Breakdown', fontweight='bold', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(clean_benchmarks, rotation=45, ha='right')

        legend = ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_alpha(0.8)

        ax1.grid(True, alpha=0.3)

        all_times = np.concatenate([user_time, system_time])
        if len(all_times) > 0 and np.max(all_times) > 0 and np.min(all_times[all_times > 0]) > 0:
            ratio = np.max(all_times) / np.min(all_times[all_times > 0])
            if ratio > 10:
                ax1.set_yscale('log')
                ax1.set_ylabel('Time (seconds) - Log Scale', fontweight='bold')

        for i, (u, s) in enumerate(zip(user_time, system_time)):
            total = u + s
            efficiency = (u / total * 100) if total > 0 else 0
            ax1.text(i, max(u, s) * 1.1, f"Total: {total:.1f}s\n({efficiency:.0f}% user)",
                     ha='center', va='bottom', fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

        memory_mb = resource_data['max_rss_kb'].values / 1024
        bars = ax2.bar(clean_benchmarks, memory_mb, alpha=0.8, color=sns.color_palette("viridis", len(memory_mb)))
        ax2.set_xlabel('Benchmark', fontweight='bold')
        ax2.set_ylabel('Peak Memory Usage (MB)', fontweight='bold')
        ax2.set_title('Peak Memory Usage', fontweight='bold', fontsize=14)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        if len(memory_mb) > 0 and np.max(memory_mb) > 0 and np.min(memory_mb[memory_mb > 0]) > 0:
            ratio = np.max(memory_mb) / np.min(memory_mb[memory_mb > 0])
            if ratio > 10:
                ax2.set_yscale('log')
                ax2.set_ylabel('Peak Memory Usage (MB) - Log Scale', fontweight='bold')

        for bar, mem in zip(bars, memory_mb):
            height = bar.get_height()
            offset = height * 0.1 if ax2.get_yscale() == 'log' else 0
            ax2.annotate(f'{mem:.1f}MB',
                        xy=(bar.get_x() + bar.get_width()/2, height + offset),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

        total_time = resource_data['elapsed_time'].values
        cpu_time = user_time + system_time
        cpu_efficiency = (cpu_time / total_time) * 100

        colors = plt.cm.RdYlGn(cpu_efficiency / 100)  # Normalize to 0-1 range
        eff_bars = ax3.bar(clean_benchmarks, cpu_efficiency, alpha=0.8, color=colors)
        ax3.set_xlabel('Benchmark', fontweight='bold')
        ax3.set_ylabel('CPU Efficiency (%)', fontweight='bold')
        ax3.set_title('CPU Utilization Efficiency', fontweight='bold', fontsize=14)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.set_ylim(0, 110)
        ax3.grid(True, alpha=0.3)

        for bar, eff in zip(eff_bars, cpu_efficiency):
            height = bar.get_height()
            ax3.annotate(f'{eff:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

        ax3.axhspan(0, 25, alpha=0.1, color='red', label='Poor')
        ax3.axhspan(25, 50, alpha=0.1, color='orange', label='Fair')
        ax3.axhspan(50, 75, alpha=0.1, color='yellow', label='Good')
        ax3.axhspan(75, 100, alpha=0.1, color='green', label='Excellent')

        rating_handles = [
            plt.Rectangle((0,0),1,1, alpha=0.3, color='red'),
            plt.Rectangle((0,0),1,1, alpha=0.3, color='orange'),
            plt.Rectangle((0,0),1,1, alpha=0.3, color='yellow'),
            plt.Rectangle((0,0),1,1, alpha=0.3, color='green')
        ]
        ax3.legend(rating_handles, ['Poor (<25%)', 'Fair (25-50%)',
                                    'Good (50-75%)', 'Excellent (>75%)'],
                  loc='upper right', frameon=True, fancybox=True)

        if len(resource_data) > 1:
            metrics_df = pd.DataFrame({
                'Benchmark': clean_benchmarks,
                'Execution Time (s)': resource_data['elapsed_time'].values,
                'Memory Usage (MB)': resource_data['max_rss_kb'].values / 1024,
                'CPU Utilization (%)': (user_time + system_time) / resource_data['elapsed_time'] * 100
            })

            best_exec_time = metrics_df['Execution Time (s)'].min()
            best_memory = metrics_df['Memory Usage (MB)'].min()
            best_cpu = metrics_df['CPU Utilization (%)'].max()

            metrics_df['Time Score'] = best_exec_time / metrics_df['Execution Time (s)'] * 100
            metrics_df['Memory Score'] = best_memory / metrics_df['Memory Usage (MB)'] * 100
            metrics_df['CPU Score'] = metrics_df['CPU Utilization (%)'] / best_cpu * 100

            metrics_df['Overall Score'] = (
                metrics_df['Time Score'] * 0.5 +
                metrics_df['Memory Score'] * 0.3 +
                metrics_df['CPU Score'] * 0.2
            )

            metrics_df = metrics_df.sort_values('Overall Score', ascending=False)

            heatmap_data = metrics_df.set_index('Benchmark')[['Time Score', 'Memory Score', 'CPU Score', 'Overall Score']]
            sns.heatmap(
                heatmap_data,
                annot=True,
                cmap='RdYlGn',
                linewidths=1,
                ax=ax4,
                vmin=0,
                vmax=100,
                fmt='.1f',
                cbar_kws={'label': 'Score (higher is better)'}
            )

            ax4.set_title('Performance Scores (100 = best)', fontweight='bold', fontsize=14)
            ax4.set_ylabel('Benchmark', fontweight='bold')

            best_idx = metrics_df['Overall Score'].idxmax()
            best_bench = metrics_df.loc[best_idx, 'Benchmark']
            ax4.text(0.5, -0.12, f"Best Overall: {best_bench} ({metrics_df.loc[best_idx, 'Overall Score']:.1f}/100)",
                    transform=ax4.transAxes, ha='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="gold", ec="orange", alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for comparison',
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Performance Comparison (Insufficient Data)')

        plt.tight_layout()
        plt.savefig(self.output_dir / "resource_usage.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Generated resource_usage.png")

        del resource_data
        gc.collect()


    def plot_performance_counters(self):
        if not self.perf_counter_paths:
            print("No performance counter data available for plotting.")
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

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

            for bench_name in self.perf_counter_paths:
                data = self._get_perf_data(bench_name)
                if data is not None:
                    value = self._get_perf_metric_value(data, metric)
                    if value is not None:
                        benchmark_names.append(bench_name)
                        values.append(value)
                    del data

            if benchmark_names and values:
                clean_names, name_mapping = self._get_clean_benchmark_names(benchmark_names)

                cmap = plt.cm.viridis
                norm = plt.Normalize(min(values), max(values))
                colors = [cmap(norm(value)) for value in values]

                bars = ax.bar(clean_names, values, alpha=0.8, color=colors, edgecolor='black', linewidth=0.5)
                ax.set_xlabel('Benchmark', fontweight='bold')
                ax.set_ylabel('Count', fontweight='bold')
                ax.set_title(f'{metric.replace("-", " ").title()}', fontweight='bold', fontsize=14)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.grid(True, axis='y', alpha=0.3)

                if len(values) > 0 and max(values) > 0 and min([v for v in values if v > 0]) > 0:
                    positive_values = [v for v in values if v > 0]
                    if positive_values:
                        ratio = max(positive_values) / min(positive_values)
                        if ratio > 100:
                            ax.set_yscale('log')
                            ax.set_ylabel('Count - Log Scale', fontweight='bold')

                if len(values) <= 8:
                    for bar, val in zip(bars, values):
                        height = bar.get_height()
                        if val >= 1e9:
                            label = f'{val/1e9:.2f}B'
                        elif val >= 1e6:
                            label = f'{val/1e6:.2f}M'
                        elif val >= 1e3:
                            label = f'{val/1e3:.2f}K'
                        else:
                            label = f'{val:.0f}'

                        offset = height * 0.1 if ax.get_yscale() == 'log' else 0
                        ax.annotate(label,
                                        xy=(bar.get_x() + bar.get_width()/2, height + offset),
                                        xytext=(0, 3),
                                        textcoords="offset points",
                                        ha='center', va='bottom', fontsize=9,
                                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

                avg_val = np.mean(values)
                ax.axhline(y=avg_val, color='red', linestyle='--', alpha=0.7,
                               label=f'Avg: {avg_val:.1e}')
                ax.legend(loc='best', frameon=True, fontsize=9)
            else:
                ax.text(0.5, 0.5, f'No {metric} data',
                        transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{metric.replace("-", " ").title()} (No Data)')

        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_counters.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Generated performance_counters.png")
        gc.collect()


    def plot_context_switches(self):
        if not self.context_stats_paths:
            print("No context switch data available for plotting.")
            return

        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import FuncFormatter

        colors = ['#3274A1', '#E1812C', '#3A923A', '#C03D3E']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        benchmarks = []
        context_switches = []
        cpu_migrations = []
        page_faults = []
        task_clock = []

        for bench_name in self.context_stats_paths:
            data = self._get_context_stats(bench_name)
            if data is not None:
                benchmarks.append(bench_name)
                context_switches.append(self._get_perf_metric_value(data, 'context-switches') or 0)
                cpu_migrations.append(self._get_perf_metric_value(data, 'cpu-migrations') or 0)
                page_faults.append(self._get_perf_metric_value(data, 'page-faults') or 0)
                tc_value = self._get_perf_metric_value(data, 'task-clock')
                task_clock.append(tc_value or 0)
                del data

        if not benchmarks:
            print("No context switch data found for plotting.")
            return

        clean_names, name_mapping = self._get_clean_benchmark_names(benchmarks)

        use_log_cs = False
        if context_switches and max(context_switches) > 0 and min([cs for cs in context_switches if cs > 0]) > 0:
            positive_cs = [cs for cs in context_switches if cs > 0]
            if positive_cs:
                ratio = max(positive_cs) / min(positive_cs)
                use_log_cs = ratio > 100

        def format_count(x, pos):
            if x >= 1e6:
                return f'{x*1e-6:.1f}M'
            elif x >= 1e3:
                return f'{x*1e-3:.1f}K'
            else:
                return f'{x:.0f}'

        formatter = FuncFormatter(format_count)

        bars1 = ax1.bar(clean_names, context_switches, alpha=0.8, color=colors[0], edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Benchmark', fontweight='bold')
        ax1.set_ylabel('Context Switches', fontweight='bold')
        ax1.set_title('Context Switches per Benchmark', fontweight='bold', fontsize=14)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(formatter)

        if use_log_cs:
            ax1.set_yscale('log')
            ax1.set_ylabel('Context Switches - Log Scale', fontweight='bold')

        avg_cs = np.mean(context_switches)
        ax1.axhline(y=avg_cs, color='red', linestyle='--', alpha=0.7, label=f'Average: {avg_cs:.0f}')
        ax1.legend(loc='best')

        use_log_mig = False
        if cpu_migrations and max(cpu_migrations) > 0 and min([m for m in cpu_migrations if m > 0]) > 0:
            positive_mig = [m for m in cpu_migrations if m > 0]
            if positive_mig:
                ratio = max(positive_mig) / min(positive_mig)
                use_log_mig = ratio > 100

        bars2 = ax2.bar(clean_names, cpu_migrations, alpha=0.8, color=colors[1], edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Benchmark', fontweight='bold')
        ax2.set_ylabel('CPU Migrations', fontweight='bold')
        ax2.set_title('CPU Migrations per Benchmark', fontweight='bold', fontsize=14)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(formatter)

        if use_log_mig:
            ax2.set_yscale('log')
            ax2.set_ylabel('CPU Migrations - Log Scale', fontweight='bold')

        if len(cpu_migrations) <= 8:
            for bar, val in zip(bars2, cpu_migrations):
                height = bar.get_height()
                offset = height * 0.1 if use_log_mig else 0
                ax2.annotate(f'{val:,.0f}',
                            xy=(bar.get_x() + bar.get_width()/2, height + offset),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

        use_log_pf = False
        if page_faults and max(page_faults) > 0 and min([pf for pf in page_faults if pf > 0]) > 0:
            positive_pf = [pf for pf in page_faults if pf > 0]
            if positive_pf:
                ratio = max(positive_pf) / min(positive_pf)
                use_log_pf = ratio > 100

        bars3 = ax3.bar(clean_names, page_faults, alpha=0.8, color=colors[2], edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('Benchmark', fontweight='bold')
        ax3.set_ylabel('Page Faults', fontweight='bold')
        ax3.set_title('Page Faults per Benchmark', fontweight='bold', fontsize=14)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(formatter)

        if use_log_pf:
            ax3.set_yscale('log')
            ax3.set_ylabel('Page Faults - Log Scale', fontweight='bold')

        if any(tc > 0 for tc in task_clock):
            cs_per_sec = [cs / (tc / 1000) if tc > 0 else 0
                        for cs, tc in zip(context_switches, task_clock)]

            use_log_rate = False
            if cs_per_sec and max(cs_per_sec) > 0 and min([rate for rate in cs_per_sec if rate > 0]) > 0:
                positive_rates = [rate for rate in cs_per_sec if rate > 0]
                if positive_rates:
                    ratio = max(positive_rates) / min(positive_rates)
                    use_log_rate = ratio > 100

            norm = plt.Normalize(min(cs_per_sec), max(cs_per_sec))
            colors_rate = plt.cm.RdYlGn_r(norm(cs_per_sec))

            bars4 = ax4.bar(clean_names, cs_per_sec, alpha=0.8, color=colors_rate, edgecolor='black', linewidth=0.5)
            ax4.set_xlabel('Benchmark', fontweight='bold')
            ax4.set_ylabel('Context Switches/sec', fontweight='bold')
            ax4.set_title('Context Switch Rate (lower is generally better)', fontweight='bold', fontsize=14)
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)

            if use_log_rate:
                ax4.set_yscale('log')
                ax4.set_ylabel('Context Switches/sec - Log Scale', fontweight='bold')

            if len(cs_per_sec) <= 8:
                for bar, val in zip(bars4, cs_per_sec):
                    height = bar.get_height()
                    offset = height * 0.1 if use_log_rate else 0
                    ax4.annotate(f'{val:,.1f}/s',
                                xy=(bar.get_x() + bar.get_width()/2, height + offset),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

            median_rate = np.median(cs_per_sec)
            ax4.axhline(y=median_rate, color='blue', linestyle='--', alpha=0.7,
                        label=f'Median: {median_rate:.1f}/s')
            ax4.legend(loc='best')
        else:
            ax4.text(0.5, 0.5, 'No timing data for rate calculation',
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Context Switch Rate (No Data)')

        plt.tight_layout()
        plt.savefig(self.output_dir / "context_switches.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Generated context_switches.png")
        gc.collect()


    def plot_scheduling_latency(self):
        if not self.sched_latency_paths:
            print("No scheduling latency data available for plotting.")
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        num_benchmarks = len(self.sched_latency_paths)
        cols = min(2, num_benchmarks)
        rows = (num_benchmarks + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        latency_summary_data = []
        benchmark_names = list(self.sched_latency_paths.keys())
        clean_benchmark_names, name_mapping = self._get_clean_benchmark_names(benchmark_names)

        for i, bench_name in enumerate(self.sched_latency_paths):
            if i >= len(axes):
                break

            ax = axes[i]
            data = self._get_sched_latency(bench_name)
            clean_name = clean_benchmark_names[i]

            if data is not None and not data.empty and 'avg_delay_ms' in data.columns and 'max_delay_ms' in data.columns:
                latency_summary_data.append({
                    'bench_name': bench_name,
                    'clean_name': clean_name,
                    'avg_latency': data['avg_delay_ms'].mean(),
                    'max_latency': data['max_delay_ms'].max(),
                    'total_switches': data['switches'].sum(),
                    'worst_task': data.loc[data['max_delay_ms'].idxmax()]['task'],
                    'worst_latency': data['max_delay_ms'].max()
                })

                sizes = data['switches'].values
                sizes = 20 + (sizes - sizes.min()) / (sizes.max() - sizes.min() + 0.1) * 200

                cmap = plt.cm.viridis_r  # Reversed so higher values (worse) are more visible
                norm = plt.Normalize(data['max_delay_ms'].min(), data['max_delay_ms'].max())
                colors = cmap(norm(data['max_delay_ms']))

                scatter = ax.scatter(data['avg_delay_ms'], data['max_delay_ms'],
                                    s=sizes, alpha=0.8, c=colors,
                                    edgecolor='black', linewidth=0.5)

                ax.set_xlabel('Average Delay (ms)', fontweight='bold')
                ax.set_ylabel('Maximum Delay (ms)', fontweight='bold')
                ax.set_title(f'{clean_name} - Scheduling Latency', fontweight='bold', fontsize=13)
                ax.grid(True, alpha=0.3)

                avg_delays = data['avg_delay_ms'].values
                max_delays = data['max_delay_ms'].values
                all_delays = np.concatenate([avg_delays[avg_delays > 0], max_delays[max_delays > 0]])

                if len(all_delays) > 0:
                    ratio = np.max(all_delays) / np.min(all_delays)
                    if ratio > 100:
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                        ax.set_xlabel('Average Delay (ms) - Log Scale', fontweight='bold')
                        ax.set_ylabel('Maximum Delay (ms) - Log Scale', fontweight='bold')

                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Maximum Delay (ms)', fontweight='bold')

                if len(data) > 1:
                    try:
                        from scipy import stats

                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            data['avg_delay_ms'], data['max_delay_ms'])

                        x_sorted = np.sort(data['avg_delay_ms'])
                        y_pred = intercept + slope * x_sorted

                        ax.plot(x_sorted, y_pred, "r--", linewidth=2, alpha=0.8,
                                label=f"Trend: slope={slope:.2f}, R²={r_value**2:.3f}")

                        n = len(data['avg_delay_ms'])
                        if n > 2:
                            mean_x = np.mean(data['avg_delay_ms'])
                            sum_square_x = np.sum((data['avg_delay_ms'] - mean_x)**2)

                            y_actual = data['max_delay_ms']
                            y_model = intercept + slope * data['avg_delay_ms']
                            residuals = y_actual - y_model
                            std_residuals = np.std(residuals)

                            t_value = stats.t.ppf(0.975, n-2)
                            conf_interval = t_value * std_residuals * np.sqrt(
                                1/n + (x_sorted - mean_x)**2 / sum_square_x)

                            ax.fill_between(x_sorted, y_pred - conf_interval, y_pred + conf_interval,
                                         alpha=0.2, color='red', label='95% Confidence')

                        ax.legend(loc='best')
                    except Exception as e:
                        if self.debug_mode:
                            print(f"Error calculating trend line: {e}")

                # Highlight tasks with highest latency
                top_latency_tasks = data.nlargest(3, 'max_delay_ms')
                for _, task in top_latency_tasks.iterrows():
                    task_name = task['task']
                    if len(task_name) > 20:
                        task_name = task_name[:17] + '...'
                    ax.annotate(task_name,
                                (task['avg_delay_ms'], task['max_delay_ms']),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=9, fontweight='bold', alpha=0.8,
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

                stats_text = f"Tasks: {len(data)}\n"
                stats_text += f"Switches: {data['switches'].sum():,}\n"
                stats_text += f"Avg Delay: {data['avg_delay_ms'].mean():.2f} ms\n"
                stats_text += f"Max Delay: {data['max_delay_ms'].max():.2f} ms"

                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, f'No latency data for\n{clean_name}',
                    transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{clean_name} - No Data')

            del data

        for i in range(len(self.sched_latency_paths), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / "scheduling_latency.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Generated scheduling_latency.png")

        if latency_summary_data:
            self._plot_latency_summary(latency_summary_data)
            self._plot_task_latency_distribution()

        gc.collect()


    def _plot_latency_summary(self, summary_data):
        if not summary_data:
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))

        clean_names = [data['clean_name'] for data in summary_data]
        avg_latencies = [data['avg_latency'] for data in summary_data]
        max_latencies = [data['max_latency'] for data in summary_data]
        total_switches = [data['total_switches'] for data in summary_data]
        worst_task_latencies = [data['worst_latency'] for data in summary_data]
        worst_tasks = [data['worst_task'] for data in summary_data]

        use_log_avg = False
        if avg_latencies and max(avg_latencies) > 0 and min([l for l in avg_latencies if l > 0]) > 0:
            positive_avg = [l for l in avg_latencies if l > 0]
            if positive_avg:
                ratio = max(positive_avg) / min(positive_avg)
                use_log_avg = ratio > 10

        color_map = plt.cm.RdYlGn_r
        norm = plt.Normalize(min(avg_latencies), max(avg_latencies))
        colors1 = [color_map(norm(val)) for val in avg_latencies]

        bars1 = ax1.bar(clean_names, avg_latencies, alpha=0.8, color=colors1, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Benchmark', fontweight='bold')
        ax1.set_ylabel('Average Scheduling Latency (ms)', fontweight='bold')
        ax1.set_title('Average Scheduling Latency Comparison', fontweight='bold', fontsize=14)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        if use_log_avg:
            ax1.set_yscale('log')
            ax1.set_ylabel('Average Scheduling Latency (ms) - Log Scale', fontweight='bold')

        for bar, val in zip(bars1, avg_latencies):
            height = bar.get_height()
            offset = height * 0.1 if use_log_avg else 0
            ax1.annotate(f'{val:.3f}ms',
                        xy=(bar.get_x() + bar.get_width()/2, height + offset),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

        use_log_worst = False
        if worst_task_latencies and max(worst_task_latencies) > 0 and min([l for l in worst_task_latencies if l > 0]) > 0:
            positive_worst = [l for l in worst_task_latencies if l > 0]
            if positive_worst:
                ratio = max(positive_worst) / min(positive_worst)
                use_log_worst = ratio > 10

        norm2 = plt.Normalize(min(worst_task_latencies), max(worst_task_latencies))
        colors2 = [color_map(norm2(val)) for val in worst_task_latencies]

        bars2 = ax2.bar(clean_names, worst_task_latencies, alpha=0.8, color=colors2, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Benchmark', fontweight='bold')
        ax2.set_ylabel('Worst Task Latency (ms)', fontweight='bold')
        ax2.set_title('Worst Task Scheduling Latency', fontweight='bold', fontsize=14)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        if use_log_worst:
            ax2.set_yscale('log')
            ax2.set_ylabel('Worst Task Latency (ms) - Log Scale', fontweight='bold')

        for i, (bar, val, task) in enumerate(zip(bars2, worst_task_latencies, worst_tasks)):
            height = bar.get_height()
            task_short = task[:10] + '...' if len(task) > 10 else task
            offset = height * 0.1 if use_log_worst else 0
            ax2.annotate(f'{val:.3f}ms\n{task_short}',
                       xy=(bar.get_x() + bar.get_width()/2, height + offset),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        use_log_switches = False
        if total_switches and max(total_switches) > 0 and min([s for s in total_switches if s > 0]) > 0:
            positive_switches = [s for s in total_switches if s > 0]
            if positive_switches:
                ratio = max(positive_switches) / min(positive_switches)
                use_log_switches = ratio > 100

        norm3 = plt.Normalize(min(total_switches), max(total_switches))
        colors3 = plt.cm.viridis(norm3(total_switches))

        bars3 = ax3.bar(clean_names, total_switches, alpha=0.8, color=colors3, edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('Benchmark', fontweight='bold')
        ax3.set_ylabel('Total Context Switches', fontweight='bold')
        ax3.set_title('Total Context Switches', fontweight='bold', fontsize=14)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)

        if use_log_switches:
            ax3.set_yscale('log')
            ax3.set_ylabel('Total Context Switches - Log Scale', fontweight='bold')

        for bar, val in zip(bars3, total_switches):
            height = bar.get_height()
            offset = height * 0.1 if use_log_switches else 0
            ax3.annotate(f'{val:,}',
                       xy=(bar.get_x() + bar.get_width()/2, height + offset),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

        scatter = ax4.scatter(total_switches, worst_task_latencies,
                            alpha=0.8, s=80, c=avg_latencies,
                            cmap='viridis', edgecolor='black', linewidth=1)
        ax4.set_xlabel('Total Context Switches', fontweight='bold')
        ax4.set_ylabel('Worst Task Latency (ms)', fontweight='bold')
        ax4.set_title('Latency vs Context Switches', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3)

        if use_log_switches:
            ax4.set_xscale('log')
            ax4.set_xlabel('Total Context Switches - Log Scale', fontweight='bold')
        if use_log_worst:
            ax4.set_yscale('log')
            ax4.set_ylabel('Worst Task Latency (ms) - Log Scale', fontweight='bold')

        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Average Latency (ms)', fontweight='bold')

        for i, clean_name in enumerate(clean_names):
            label = clean_name[:12] + '...' if len(clean_name) > 12 else clean_name
            ax4.annotate(label,
                        (total_switches[i], worst_task_latencies[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

        if len(clean_names) > 2:
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    np.log10(total_switches) if use_log_switches else total_switches,
                    np.log10(worst_task_latencies) if use_log_worst else worst_task_latencies
                )

                x_range = np.linspace(min(total_switches), max(total_switches), 100)
                if use_log_switches and use_log_worst:
                    y_pred = 10**(slope * np.log10(x_range) + intercept)
                elif use_log_worst:
                    y_pred = 10**(slope * x_range + intercept)
                elif use_log_switches:
                    y_pred = slope * np.log10(x_range) + intercept
                else:
                    y_pred = slope * x_range + intercept

                ax4.plot(x_range, y_pred, 'r--', alpha=0.7,
                        label=f'Trend (R²: {r_value**2:.2f})')
                ax4.legend(loc='best')
            except Exception as e:
                if self.debug_mode:
                    print(f"Could not calculate trend line: {e}")

        plt.tight_layout()
        plt.savefig(self.output_dir / "latency_summary.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Generated latency_summary.png")


    def _plot_task_latency_distribution(self):
        if not self.sched_latency_paths:
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        latency_data = []
        benchmark_names = list(self.sched_latency_paths.keys())
        clean_names, name_mapping = self._get_clean_benchmark_names(benchmark_names)

        for i, bench_name in enumerate(self.sched_latency_paths):
            data = self._get_sched_latency(bench_name)
            clean_name = clean_names[i]
            if data is not None and not data.empty:
                top_tasks = data.nlargest(5, 'max_delay_ms')
                for _, row in top_tasks.iterrows():
                    latency_data.append({
                        'benchmark': bench_name,
                        'clean_benchmark': clean_name,
                        'task': row['task'],
                        'avg_delay': row['avg_delay_ms'],
                        'max_delay': row['max_delay_ms'],
                        'switches': row['switches']
                    })
                del data

        if not latency_data:
            return

        latency_df = pd.DataFrame(latency_data)

        top_latency_tasks = latency_df.nlargest(10, 'max_delay')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})

        task_labels = [f"{row['task'][:15]}...\n({row['clean_benchmark']})" if len(row['task']) > 15
                      else f"{row['task']}\n({row['clean_benchmark']})"
                      for _, row in top_latency_tasks.iterrows()]

        x = np.arange(len(task_labels))
        width = 0.35

        avg_delays = top_latency_tasks['avg_delay'].values
        max_delays = top_latency_tasks['max_delay'].values
        switches = top_latency_tasks['switches'].values

        all_delays = np.concatenate([avg_delays[avg_delays > 0], max_delays[max_delays > 0]])
        use_log_task = False
        if len(all_delays) > 0:
            ratio = np.max(all_delays) / np.min(all_delays)
            use_log_task = ratio > 10

        bar1 = ax1.bar(x - width/2, avg_delays, width, label='Avg Delay (ms)',
                      alpha=0.8, color='#3274A1', edgecolor='black', linewidth=0.5)
        bar2 = ax1.bar(x + width/2, max_delays, width, label='Max Delay (ms)',
                      alpha=0.8, color='#E1812C', edgecolor='black', linewidth=0.5)

        ax1.set_ylabel('Delay (ms)', fontweight='bold')
        ax1.set_title('Top Tasks by Scheduling Latency', fontweight='bold', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(task_labels, rotation=45, ha='right')
        ax1.legend(loc='best', frameon=True, fancybox=True)
        ax1.grid(True, axis='y', alpha=0.3)

        if use_log_task:
            ax1.set_yscale('log')
            ax1.set_ylabel('Delay (ms) - Log Scale', fontweight='bold')

        for i, (max_val, avg_val) in enumerate(zip(max_delays, avg_delays)):
            height = max_val
            offset = height * 0.1 if use_log_task else max_val * 0.05
            ax1.annotate(f'{max_val:.2f}ms',
                       xy=(i + width/2, height + offset),
                       ha='center', va='bottom', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

            ratio = max_val / avg_val if avg_val > 0 else float('inf')
            ax1.annotate(f'{ratio:.1f}x',
                       xy=(i, avg_val * 1.1),
                       ha='center', va='bottom', fontsize=8,
                       color='blue')

        for i, switches_val in enumerate(switches):
            ax1.text(i, 0.1 if not use_log_task else min(all_delays) * 0.1,
                    f"{switches_val:,} sw",
                    ha='center', va='bottom', fontsize=9, rotation=90, alpha=0.7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

        task_benchmark_matrix = pd.crosstab(
            latency_df['task'], latency_df['clean_benchmark'],
            values=latency_df['max_delay'], aggfunc='max'
        ).fillna(0)

        if not task_benchmark_matrix.empty and task_benchmark_matrix.shape[0] > 1:
            task_totals = task_benchmark_matrix.sum(axis=1)
            sorted_tasks = task_totals.sort_values(ascending=False).index[:15]

            plot_matrix = task_benchmark_matrix.loc[sorted_tasks]

            sns.heatmap(plot_matrix, cmap='viridis', linewidths=0.5,
                      annot=True, fmt='.1f', ax=ax2, cbar_kws={'label': 'Max Delay (ms)'})

            ax2.set_title('Task Latency by Benchmark', fontweight='bold', fontsize=14)
            ax2.set_ylabel('Task Name', fontweight='bold')
            ax2.set_xlabel('Benchmark', fontweight='bold')

            task_labels = [t[:25] + '...' if len(t) > 25 else t for t in plot_matrix.index]
            ax2.set_yticklabels(task_labels, rotation=0)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        else:
            ax2.text(0.5, 0.5, 'Insufficient data for task/benchmark heatmap',
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / "task_latency_distribution.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Generated task_latency_distribution.png")

        del latency_df, top_latency_tasks
        gc.collect()


    @_with_memory_limit
    def plot_correlation_analysis(self):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        benchmarks = set(list(self.hyperfine_paths.keys()) +
                        list(self.perf_counter_paths.keys()) +
                        list(self.context_stats_paths.keys()))

        if len(benchmarks) > 20:
            print("Too many benchmarks for correlation analysis, limiting to 20")
            benchmarks = list(benchmarks)[:20]

        correlation_data = []

        for bench_name in benchmarks:
            row = {'benchmark': bench_name}

            if bench_name in self.hyperfine_paths:
                hyper_data = self._load_hyperfine_data_for_bench(bench_name)
                if 'json' in hyper_data:
                    results = hyper_data['json'].get('results', [])
                    if results:
                        result = results[0]
                        row['execution_time'] = result.get('mean', 0)
                        row['execution_std'] = result.get('stddev', 0)
                del hyper_data

            resource_data = self._load_resource_data()
            if resource_data is not None:
                resource_row = resource_data[
                    resource_data['benchmark'] == bench_name
                ]
                if not resource_row.empty:
                    row['user_time'] = resource_row.iloc[0]['user_time']
                    row['system_time'] = resource_row.iloc[0]['system_time']
                    row['memory_mb'] = resource_row.iloc[0]['max_rss_kb'] / 1024
                del resource_row
            del resource_data

            if bench_name in self.perf_counter_paths:
                perf_data = self._get_perf_data(bench_name)
                if perf_data is not None:
                    row['cycles'] = self._get_perf_metric_value(perf_data, 'cycles')
                    row['instructions'] = self._get_perf_metric_value(perf_data, 'instructions')
                    row['cache_references'] = self._get_perf_metric_value(perf_data, 'cache-references')
                    row['cache_misses'] = self._get_perf_metric_value(perf_data, 'cache-misses')

                    if row.get('cycles') and row.get('instructions'):
                        row['ipc'] = row['instructions'] / row['cycles']

                    if row.get('cache_references') and row.get('cache_misses'):
                        row['cache_miss_rate'] = row['cache_misses'] / row['cache_references'] * 100

                    del perf_data

            if bench_name in self.context_stats_paths:
                context_data = self._get_context_stats(bench_name)
                if context_data is not None:
                    row['context_switches'] = self._get_perf_metric_value(context_data, 'context-switches')
                    row['cpu_migrations'] = self._get_perf_metric_value(context_data, 'cpu-migrations')
                    del context_data

            if bench_name in self.sched_latency_paths:
                sched_data = self._get_sched_latency(bench_name)
                if sched_data is not None and not sched_data.empty:
                    row['avg_sched_latency'] = sched_data['avg_delay_ms'].mean()
                    row['max_sched_latency'] = sched_data['max_delay_ms'].max()
                    row['total_switches'] = sched_data['switches'].sum()
                    del sched_data

            correlation_data.append(row)

            gc.collect()

        if len(correlation_data) > 1:
            df = pd.DataFrame(correlation_data)

            if 'benchmark' in df.columns:
                clean_names, name_mapping = self._get_clean_benchmark_names(df['benchmark'].tolist())
                df['clean_benchmark'] = clean_names

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'benchmark']

            if len(numeric_cols) > 1:
                corr_df = df[numeric_cols].dropna(axis=1, how='all')

                pretty_names = {
                    'execution_time': 'Execution Time (s)',
                    'execution_std': 'Time Std Dev (s)',
                    'user_time': 'User CPU Time (s)',
                    'system_time': 'System CPU Time (s)',
                    'memory_mb': 'Memory Usage (MB)',
                    'cycles': 'CPU Cycles',
                    'instructions': 'Instructions',
                    'cache_references': 'Cache References',
                    'cache_misses': 'Cache Misses',
                    'ipc': 'Instructions Per Cycle',
                    'cache_miss_rate': 'Cache Miss Rate (%)',
                    'context_switches': 'Context Switches',
                    'cpu_migrations': 'CPU Migrations',
                    'avg_sched_latency': 'Avg Sched Latency (ms)',
                    'max_sched_latency': 'Max Sched Latency (ms)',
                    'total_switches': 'Total Task Switches'
                }

                corr_df = corr_df.rename(columns={col: pretty_names.get(col, col) for col in corr_df.columns})

                if corr_df.shape[1] > 1:
                    fig, ax = plt.subplots(figsize=(14, 12))

                    try:
                        corr_df = corr_df.apply(pd.to_numeric, errors='coerce')
                        correlation_matrix = corr_df.corr()

                        mask = np.zeros_like(correlation_matrix)
                        mask[np.triu_indices_from(mask)] = True

                        n = len(df)
                        if n > 3:
                            r_crit = 1.96 / np.sqrt(n)
                            sig_mask = np.abs(correlation_matrix) > r_crit
                            annot = np.empty_like(correlation_matrix, dtype=object)
                            for i in range(correlation_matrix.shape[0]):
                                for j in range(correlation_matrix.shape[1]):
                                    val = correlation_matrix.iloc[i, j]
                                    star = '*' if sig_mask.iloc[i, j] and i != j else ''
                                    if pd.notnull(val) and isinstance(val, (int, float, np.number)):
                                        annot[i, j] = f'{val:.2f}{star}'
                                    else:
                                        annot[i, j] = 'NA'
                        else:
                            annot = True

                        sns.heatmap(correlation_matrix, annot=annot, fmt='', cmap='RdBu_r', center=0,
                                   mask=mask, square=True, linewidths=.5,
                                   cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
                                   ax=ax, vmin=-1, vmax=1)

                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

                        ax.set_title('Correlation Matrix - Performance Metrics', fontweight='bold', fontsize=16, pad=20)

                        if n > 3:
                            plt.figtext(0.5, 0.01,
                                       "* indicates statistically significant correlation (p < 0.05)",
                                       ha="center", fontsize=10,
                                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

                        plt.tight_layout()
                        plt.savefig(self.output_dir / "correlation_analysis.png", dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        print("✓ Generated correlation_analysis.png")

                        self._generate_scatter_matrix(df, pretty_names)

                        del correlation_matrix
                    except Exception as e:
                        print(f"Error in correlation analysis: {e}")
                        if self.debug_mode:
                            import traceback
                            traceback.print_exc()
                else:
                    print("Insufficient numeric data for correlation analysis.")

                del corr_df
            else:
                print("Insufficient numeric data for correlation analysis.")

            del df
        else:
            print("Insufficient benchmarks for correlation analysis.")

        del correlation_data
        gc.collect()


    def _generate_scatter_matrix(self, df, pretty_names):
        import matplotlib.pyplot as plt
        import seaborn as sns

        key_metrics = ['execution_time', 'memory_mb', 'context_switches', 'cycles']
        key_metrics = [col for col in key_metrics if col in df.columns]

        if len(key_metrics) >= 3:
            plot_df = df.copy()
            plot_df = plot_df.rename(columns={col: pretty_names.get(col, col) for col in key_metrics})

            hue_col = 'clean_benchmark' if 'clean_benchmark' in df.columns else None

            try:
                plt.figure(figsize=(12, 10))
                scatter_matrix = sns.pairplot(
                    plot_df,
                    vars=[pretty_names.get(col, col) for col in key_metrics],
                    hue=hue_col,
                    markers='o',
                    diag_kind='kde',
                    plot_kws={'alpha': 0.8, 's': 80, 'edgecolor': 'white'},
                    diag_kws={'shade': True},
                    corner=True
                )

                scatter_matrix.fig.suptitle('Performance Metrics Relationships', fontsize=16, y=1.02)

                plt.tight_layout()
                plt.savefig(self.output_dir / "metrics_scatter_matrix.png", dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ Generated metrics_scatter_matrix.png")
            except Exception as e:
                if self.debug_mode:
                    print(f"Error generating scatter matrix: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            if self.debug_mode:
                print(f"Insufficient metrics for scatter matrix: {key_metrics}")


    @_with_memory_limit
    def plot_sched_timeline(self):
        if not self.sched_timeline_paths:
            print("No scheduling timeline data available for plotting.")
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from matplotlib.colors import LinearSegmentedColormap, to_rgba
        from matplotlib.collections import LineCollection
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import Patch

        sample_rate = 1.0
        benchmark_names = list(self.sched_timeline_paths.keys())
        clean_names, name_mapping = self._get_clean_benchmark_names(benchmark_names)

        for i, bench_name in enumerate(self.sched_timeline_paths):
            clean_name = clean_names[i]

            print(f"Processing timeline for {clean_name}...")

            sample_data = self._get_sched_timeline(bench_name, sample_rate=0.01)
            if sample_data is None or sample_data.empty:
                continue

            estimated_total = len(sample_data) * 100
            if estimated_total > 50000:
                sample_rate = min(1.0, 50000 / estimated_total)
                print(f"  Large timeline dataset detected ({estimated_total:,} est. points), using {sample_rate:.1%} sample")
            del sample_data

            timeline_data = self._get_sched_timeline(bench_name, sample_rate=sample_rate)
            if timeline_data is None or timeline_data.empty:
                continue

            print(f"  Generating visualization with {len(timeline_data):,} data points...")

            fig = plt.figure(figsize=(16, 14))
            gs = gridspec.GridSpec(4, 1, height_ratios=[1, 3, 1, 1], hspace=0.3)

            ax_overview = fig.add_subplot(gs[0])
            ax_timeline = fig.add_subplot(gs[1])
            ax_delays = fig.add_subplot(gs[2])
            ax_util = fig.add_subplot(gs[3])

            base_time = timeline_data['timestamp'].min()
            max_time = timeline_data['timestamp'].max()
            timeline_duration_ms = (max_time - base_time) * 1000

            cpus = sorted(timeline_data['cpu'].unique())

            task_types = {
                'idle': [t for t in timeline_data['task_name'].unique()
                        if any(x in t.lower() for x in ['idle', 'swapper'])],
                'system': [t for t in timeline_data['task_name'].unique()
                        if any(x in t.lower() for x in ['kworker', 'migration', 'ksoftirqd', 'watchdog', 'irq'])],
                'application': []  # Will fill with other tasks
            }

            for task in timeline_data['task_name'].unique():
                if task not in task_types['idle'] and task not in task_types['system']:
                    task_types['application'].append(task)

            colors = {
                'idle': (0.8, 0.8, 0.8, 0.6),        # Light gray
                'system': (1.0, 0.5, 0.0, 0.8),      # Orange
                'application': (0.0, 0.6, 1.0, 0.9)  # Blue
            }

            total_context_switches = len(timeline_data)
            avg_delay = timeline_data['sched_delay_ms'].mean()
            max_delay = timeline_data['sched_delay_ms'].max()

            time_segments = []
            colors_list = []

            sorted_timeline = timeline_data.sort_values(['cpu', 'timestamp'])

            for cpu in cpus:
                cpu_data = sorted_timeline[sorted_timeline['cpu'] == cpu]

                if cpu_data.empty:
                    continue

                last_time = None
                last_task = None

                for _, row in cpu_data.iterrows():
                    time_ms = (row['timestamp'] - base_time) * 1000
                    task = row['task_name']

                    if task in task_types['idle']:
                        task_type = 'idle'
                    elif task in task_types['system']:
                        task_type = 'system'
                    else:
                        task_type = 'application'

                    if last_time is not None and task != last_task:
                        time_segments.append([(last_time, cpu), (time_ms, cpu)])
                        colors_list.append(colors[task_type])

                    last_time = time_ms
                    last_task = task

                del cpu_data

            if time_segments:
                timeline_collection = LineCollection(time_segments, colors=colors_list, linewidths=7, alpha=0.8)
                ax_overview.add_collection(timeline_collection)

            ax_overview.set_xlim(0, timeline_duration_ms)
            ax_overview.set_ylim(-1, len(cpus))
            ax_overview.set_yticks(cpus)
            ax_overview.set_yticklabels([f'CPU {cpu}' for cpu in cpus])
            ax_overview.set_title(f'Scheduling Timeline Overview - {clean_name}',
                                 fontweight='bold', fontsize=14, pad=10)
            ax_overview.grid(True, axis='x', alpha=0.3)

            legend_patches = [
                Patch(color=colors['idle'], label='Idle Tasks'),
                Patch(color=colors['system'], label='System Tasks'),
                Patch(color=colors['application'], label='Application Tasks')
            ]
            ax_overview.legend(handles=legend_patches, loc='upper right', ncol=3,
                             fancybox=True, shadow=True)

            task_stats = timeline_data.groupby('task_name').agg({
                'run_time_ms': ['sum', 'mean', 'count'],
                'sched_delay_ms': ['mean', 'max']
            })

            task_stats.columns = ['_'.join(col).strip() for col in task_stats.columns]

            task_stats['importance'] = (
                task_stats['run_time_ms_sum'] * 0.4 +   # Total runtime
                task_stats['run_time_ms_count'] * 0.3 + # Frequency of appearance
                task_stats['sched_delay_ms_max'] * 50 + # Max scheduling delay
                task_stats['sched_delay_ms_mean'] * 100 # Average scheduling delay
            )

            top_tasks = task_stats.sort_values('importance', ascending=False).head(15).index.tolist()

            if not any(t in task_types['idle'] for t in top_tasks):
                idle_tasks = task_stats[task_stats.index.isin(task_types['idle'])].sort_values('run_time_ms_sum', ascending=False)
                if not idle_tasks.empty:
                    top_tasks.append(idle_tasks.index[0])

            if not any(t in task_types['system'] for t in top_tasks):
                system_tasks = task_stats[task_stats.index.isin(task_types['system'])].sort_values('run_time_ms_sum', ascending=False)
                if not system_tasks.empty:
                    top_tasks.append(system_tasks.index[0])

            top_tasks = list(dict.fromkeys(top_tasks))

            cmap = plt.cm.tab20
            top_task_colors = {}

            for idx, task in enumerate(top_tasks):
                if task in task_types['idle']:
                    top_task_colors[task] = colors['idle']
                elif task in task_types['system']:
                    top_task_colors[task] = colors['system']
                else:
                    # Use the tab20 colormap for application tasks
                    top_task_colors[task] = cmap(idx % 20)

            segments = []
            segment_colors = []

            task_blocks = {cpu: [] for cpu in cpus}

            # Process events chronologically
            for cpu in cpus:
                cpu_data = timeline_data[timeline_data['cpu'] == cpu].sort_values('timestamp')

                current_blocks = {}

                for idx, row in cpu_data.iterrows():
                    time_ms = (row['timestamp'] - base_time) * 1000
                    task = row['task_name']
                    run_time = row['run_time_ms']

                    if run_time < 0.5:
                        continue

                    if task in top_tasks:
                        task_blocks[cpu].append({
                            'task': task,
                            'start': time_ms,
                            'end': time_ms + run_time,
                            'delay': row['sched_delay_ms']
                        })

                del cpu_data

            for cpu in cpus:
                y_base = cpu
                blocks = sorted(task_blocks[cpu], key=lambda x: x['start'])

                for block in blocks:
                    task = block['task']
                    start = block['start']
                    end = block['end']
                    delay = block['delay']

                    if end - start > 0.5:
                        rect = plt.Rectangle(
                            (start, y_base - 0.4),
                            end - start,
                            0.8,
                            facecolor=top_task_colors[task],
                            edgecolor='black',
                            linewidth=0.5,
                            alpha=0.8
                        )
                        ax_timeline.add_patch(rect)

                        if delay > avg_delay * 2:
                            ax_timeline.plot(
                                [start, start],
                                [y_base - 0.5, y_base + 0.5],
                                'r-',
                                linewidth=2
                            )

            for cpu in cpus:
                ax_timeline.axhspan(cpu - 0.5, cpu + 0.5, color='black', alpha=0.05)

            ax_timeline.set_xlim(0, timeline_duration_ms)
            ax_timeline.set_ylim(-1, len(cpus))
            ax_timeline.set_yticks(cpus)
            ax_timeline.set_yticklabels([f'CPU {cpu}' for cpu in cpus])
            ax_timeline.set_xlabel('Time (ms)', fontweight='bold')
            ax_timeline.grid(True, axis='x', alpha=0.3)

            time_markers = np.linspace(0, timeline_duration_ms, 10)
            for tm in time_markers:
                ax_timeline.axvline(tm, color='gray', linestyle='--', alpha=0.4)

            legend_handles = []
            for task in top_tasks:
                short_name = task[:20] + '...' if len(task) > 20 else task
                legend_handles.append(Patch(color=top_task_colors[task], label=short_name))

            legend_cols = 3 if len(top_tasks) > 6 else 2
            legend = ax_timeline.legend(handles=legend_handles, loc='upper right',
                                      ncol=legend_cols, fontsize='small',
                                      fancybox=True, framealpha=0.9)
            legend.get_frame().set_alpha(0.9)

            ax_timeline.set_title(
                f'Detailed Task Execution (showing {len(top_tasks)} key tasks, '
                f'{total_context_switches:,} events, '
                f'avg delay: {avg_delay:.2f}ms)',
                fontweight='bold', fontsize=14
            )

            delays = timeline_data['sched_delay_ms'].values
            non_zero_delays = delays[delays > 0]

            if len(non_zero_delays) > 0:
                use_log = max(non_zero_delays) / np.percentile(non_zero_delays, 50) > 10

                colors = plt.cm.viridis_r

                if use_log:
                    bins = np.logspace(
                        np.log10(max(0.01, non_zero_delays.min())),
                        np.log10(non_zero_delays.max() * 1.1),
                        30
                    )
                    ax_delays.set_xscale('log')
                    ax_delays.hist(non_zero_delays, bins=bins, alpha=0.8, color='#3274A1',
                                 edgecolor='black', linewidth=0.5)
                    ax_delays.set_xlabel('Scheduling Delay (ms) - Log Scale', fontweight='bold')
                else:
                    bins = 30
                    ax_delays.hist(non_zero_delays, bins=bins, alpha=0.8, color='#3274A1',
                                 edgecolor='black', linewidth=0.5)
                    ax_delays.set_xlabel('Scheduling Delay (ms)', fontweight='bold')

                percentiles = [50, 90, 99]
                percentile_values = np.percentile(non_zero_delays, percentiles)
                percentile_colors = ['green', 'orange', 'red']

                for pct, val, color in zip(percentiles, percentile_values, percentile_colors):
                    ax_delays.axvline(val, color=color, linestyle='--', alpha=0.8, linewidth=2,
                                    label=f'{pct}th percentile: {val:.2f} ms')

                mean_delay = non_zero_delays.mean()
                ax_delays.axvline(mean_delay, color='blue', linestyle='-', alpha=0.8, linewidth=2,
                                label=f'Mean: {mean_delay:.2f} ms')

                ax_delays.set_ylabel('Frequency', fontweight='bold')
                ax_delays.set_title('Scheduling Delay Distribution', fontweight='bold', fontsize=14)
                ax_delays.legend(loc='upper right', fancybox=True, shadow=True)
                ax_delays.grid(True, alpha=0.3)

                stats_text = (
                    f"Total samples: {len(non_zero_delays):,}\n"
                    f"Mean: {mean_delay:.2f}ms • Median: {percentile_values[0]:.2f}ms\n"
                    f"90%: {percentile_values[1]:.2f}ms • 99%: {percentile_values[2]:.2f}ms"
                )
                ax_delays.text(0.02, 0.95, stats_text, transform=ax_delays.transAxes,
                              va='top', fontsize=10,
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax_delays.text(0.5, 0.5, 'No non-zero scheduling delays found',
                              transform=ax_delays.transAxes, ha='center', va='center')
                ax_delays.set_title('Scheduling Delay Distribution (No Data)')

            time_bins = np.linspace(0, timeline_duration_ms, min(100, int(timeline_duration_ms/10) + 1))
            cpu_util = np.zeros((len(cpus), len(time_bins)-1))

            for cpu_idx, cpu in enumerate(cpus):
                cpu_data = timeline_data[timeline_data['cpu'] == cpu]

                for _, row in cpu_data.iterrows():
                    start = (row['timestamp'] - base_time) * 1000
                    duration = row['run_time_ms']
                    end = start + duration

                    if row['task_name'] in task_types['idle']:
                        continue

                    start_bin = max(0, np.searchsorted(time_bins, start) - 1)
                    end_bin = min(len(time_bins)-1, np.searchsorted(time_bins, end))

                    for b in range(start_bin, end_bin):
                        bin_start = time_bins[b]
                        bin_end = time_bins[b+1]

                        overlap_start = max(start, bin_start)
                        overlap_end = min(end, bin_end)
                        overlap_duration = overlap_end - overlap_start

                        bin_width = bin_end - bin_start
                        if bin_width > 0:
                            cpu_util[cpu_idx, b] += (overlap_duration / bin_width) * 100

                del cpu_data

            cpu_util = np.minimum(cpu_util, 100)

            im = ax_util.imshow(cpu_util, aspect='auto', interpolation='nearest',
                              extent=[0, timeline_duration_ms, -0.5, len(cpus)-0.5],
                              cmap='RdYlGn', vmin=0, vmax=100)

            ax_util.set_yticks(range(len(cpus)))
            ax_util.set_yticklabels([f'CPU {cpu}' for cpu in cpus])
            ax_util.set_xlabel('Time (ms)', fontweight='bold')
            ax_util.set_ylabel('CPU', fontweight='bold')
            ax_util.set_title('CPU Utilization Over Time (%)', fontweight='bold', fontsize=14)

            cbar = plt.colorbar(im, ax=ax_util)
            cbar.set_label('Utilization %', fontweight='bold')

            avg_util = cpu_util.mean()
            max_util = cpu_util.max()

            util_text = (
                f"Avg Utilization: {avg_util:.1f}% • Max Utilization: {max_util:.1f}% • "
                f"Duration: {timeline_duration_ms/1000:.2f}s"
            )
            ax_util.text(0.01, -0.15, util_text,
                        transform=ax_util.transAxes, ha='left', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            safe_name = clean_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            plt.savefig(self.output_dir / f"{safe_name}_sched_timeline.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Generated {safe_name}_sched_timeline.png")

            self._plot_task_timeline(bench_name, clean_name, timeline_data, top_tasks, top_task_colors)

            del timeline_data
            gc.collect()



    @_with_memory_limit
    def _plot_task_timeline(self, bench_name, clean_name, timeline_data, top_tasks, task_colors):
        if timeline_data is None or timeline_data.empty:
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from matplotlib.collections import PolyCollection
        from matplotlib.patches import Patch

        task_counts = timeline_data['task_name'].value_counts()
        task_durations = timeline_data.groupby('task_name')['run_time_ms'].sum().sort_values(ascending=False)

        significant_tasks = task_durations[task_durations > task_durations.sum() * 0.01].index.tolist()
        frequent_tasks = task_counts.head(10).index.tolist()
        important_tasks = list(set(significant_tasks + frequent_tasks))
        idle_tasks = [t for t in important_tasks
                     if any(x in t.lower() for x in ['idle', 'kworker', 'system', 'swapper', 'migration'])]
        display_tasks = important_tasks[:20]

        task_first_seen = {task: timeline_data[timeline_data['task_name'] == task]['timestamp'].min()
                          for task in display_tasks}
        display_tasks = sorted([t for t in display_tasks if t not in idle_tasks],
                             key=lambda x: task_first_seen[x]) + sorted(idle_tasks)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12),
                                     gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})

        base_time = timeline_data['timestamp'].min()
        max_time = timeline_data['timestamp'].max()
        timeline_duration_ms = (max_time - base_time) * 1000

        colors = plt.cm.tab20(np.linspace(0, 1, len(display_tasks)))
        for i, task in enumerate(display_tasks):
            if task in idle_tasks:
                colors[i] = [0.8, 0.8, 0.8, 0.7]  # Light gray for idle tasks

        task_cmap = dict(zip(display_tasks, colors))

        cpu_colors = plt.cm.Set2(np.linspace(0, 1, 8))
        cpus = sorted(timeline_data['cpu'].unique())

        y_pos = {task: i for i, task in enumerate(display_tasks)}

        segments = {task: [] for task in display_tasks}

        displayed_runtime = 0
        total_runtime = 0

        sorted_data = timeline_data.sort_values('timestamp')
        for _, row in sorted_data.iterrows():
            task = row['task_name']
            time_ms = (row['timestamp'] - base_time) * 1000
            run_time = row['run_time_ms']
            cpu = row['cpu']

            total_runtime += run_time

            if task in display_tasks and run_time > 0.1:  # Only include meaningful durations
                segments[task].append((time_ms, time_ms + run_time, cpu))
                displayed_runtime += run_time

        hidden_runtime_pct = 100 - (displayed_runtime / total_runtime * 100) if total_runtime > 0 else 0

        task_cpu_times = {task: {cpu: 0 for cpu in cpus} for task in display_tasks}

        for task in display_tasks:
            for start, end, cpu in segments[task]:
                task_cpu_times[task][cpu] += (end - start)

        idle_cpu_util = {cpu: 0 for cpu in cpus}
        for task in idle_tasks:
            if task in task_cpu_times:
                for cpu in cpus:
                    idle_cpu_util[cpu] += task_cpu_times[task][cpu] / timeline_duration_ms * 100 if timeline_duration_ms > 0 else 0

        avg_idle_pct = sum(idle_cpu_util.values()) / len(cpus) if cpus else 0

        for task_idx, task in enumerate(display_tasks):
            task_segments = segments[task]
            if not task_segments:
                continue

            y = y_pos[task]
            verts = []
            cpu_colors_list = []

            for start, end, cpu in task_segments:
                verts.append([(start, y - 0.4), (start, y + 0.4),
                            (end, y + 0.4), (end, y - 0.4)])
                cpu_colors_list.append(cpu_colors[cpu % len(cpu_colors)])

            if verts:
                if task in idle_tasks:
                    bars = PolyCollection(verts, facecolors=[0.8, 0.8, 0.8, 0.7],
                                        edgecolors='gray', linewidth=0.3, alpha=0.6)
                else:
                    bars = PolyCollection(verts, facecolors=cpu_colors_list,
                                        edgecolors='black', linewidth=0.5, alpha=0.8)
                ax1.add_collection(bars)

        ax1.set_yticks(list(y_pos.values()))

        task_labels = []
        for task in display_tasks:
            cpu_percentages = [(task_cpu_times[task][cpu] / timeline_duration_ms * 100)
                             for cpu in cpus]
            avg_cpu_util = sum(cpu_percentages) / len(cpus) if cpus else 0
            max_cpu_util = max(cpu_percentages) if cpu_percentages else 0

            display_name = task[:25] + '...' if len(task) > 25 else task

            if max_cpu_util > 50:
                label_text = f"{display_name} (avg: {avg_cpu_util:.1f}%, max: {max_cpu_util:.1f}%)"
            else:
                label_text = f"{display_name} (avg: {avg_cpu_util:.1f}%)"

            task_labels.append(label_text)

        ax1.set_yticklabels(task_labels)
        ax1.set_ylim(-1, len(display_tasks))

        ax1.set_xlim(0, timeline_duration_ms)
        ax1.grid(True, axis='x', alpha=0.3)
        ax1.set_xlabel('Time from start (ms)', fontweight='bold')

        cpu_patches = [Patch(color=cpu_colors[cpu % len(cpu_colors)], label=f'CPU {cpu}')
                      for cpu in cpus]

        title = f'Task Execution Timeline - {clean_name}\n'
        title += f'Showing {len(display_tasks)} tasks ({displayed_runtime/total_runtime*100:.1f}% of runtime)'
        if avg_idle_pct > 5:
            title += f' • Avg CPU Idle: {avg_idle_pct:.1f}%'
        if hidden_runtime_pct > 5:
            title += f' • Hidden: {hidden_runtime_pct:.1f}%'

        ax1.set_title(title, fontweight='bold', fontsize=14, pad=10)
        ax1.legend(handles=cpu_patches, loc='upper right', ncol=2,
                  fancybox=True, shadow=True)

        time_bins = np.linspace(0, timeline_duration_ms, min(200, int(timeline_duration_ms/5) + 1))

        all_activity = np.zeros((len(cpus), len(time_bins)-1))
        shown_activity = np.zeros((len(cpus), len(time_bins)-1))

        for _, row in timeline_data.iterrows():
            cpu_idx = cpus.index(row['cpu'])
            start = (row['timestamp'] - base_time) * 1000
            duration = row['run_time_ms']
            end = start + duration

            start_bin = max(0, np.searchsorted(time_bins, start) - 1)
            end_bin = min(len(time_bins)-1, np.searchsorted(time_bins, end))

            for b in range(start_bin, end_bin):
                all_activity[cpu_idx, b] += duration / (time_bins[b+1] - time_bins[b])

                if row['task_name'] in display_tasks and duration > 0.1:
                    shown_activity[cpu_idx, b] += duration / (time_bins[b+1] - time_bins[b])

        hidden_mask = (all_activity > 0.1) & (shown_activity < 0.05)

        all_activity = np.clip(all_activity / all_activity.max() if all_activity.max() > 0 else all_activity, 0, 1)

        im = ax2.imshow(all_activity, aspect='auto', interpolation='nearest',
                      extent=[0, timeline_duration_ms, -0.5, len(cpus)-0.5],
                      cmap='viridis')

        if hidden_mask.any():
            ax2.contour(hidden_mask, levels=[0.5],
                      extent=[0, timeline_duration_ms, -0.5, len(cpus)-0.5],
                      colors='red', linewidths=1.5, alpha=0.8)

        ax2.set_yticks(range(len(cpus)))
        ax2.set_yticklabels([f'CPU {cpu}' for cpu in cpus])
        ax2.set_xlabel('Time from start (ms)', fontweight='bold')
        ax2.set_ylabel('CPU', fontweight='bold')
        ax2.set_title('CPU Activity Intensity (red outline = activity not shown in timeline)',
                     fontweight='bold', fontsize=14)

        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Activity Level', fontweight='bold')

        missing_pct = ((all_activity > 0.1) & (shown_activity < 0.05)).sum() / (all_activity > 0).sum() * 100
        if missing_pct > 10:
            ax2.text(0.5, -0.2,
                    f"Note: {missing_pct:.1f}% of CPU activity involves tasks not shown in timeline\n"
                    f"(short-duration tasks, system tasks, or less frequent tasks)",
                    transform=ax2.transAxes, ha='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.tight_layout()
        safe_name = clean_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        plt.savefig(self.output_dir / f"{safe_name}_task_timeline.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Generated {safe_name}_task_timeline.png")
        gc.collect()


    def plot_ridgeline(self):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import joypy
            from matplotlib import cm

            print("Generating ridgeline plots for time distributions...")

            data_dict = {}
            for bench_name in self.hyperfine_paths:
                hyper_data = self._load_hyperfine_data_for_bench(bench_name)
                clean_name = self._clean_benchmark_name(bench_name)
                if 'json' in hyper_data and 'results' in hyper_data['json']:
                    times = hyper_data['json']['results'][0].get('times', [])
                    if times and len(times) > 3:
                        data_dict[clean_name] = times
                del hyper_data

            if len(data_dict) > 1:
                df = pd.DataFrame(data_dict)

                medians = df.median()
                sorted_cols = medians.sort_values().index.tolist()
                df = df[sorted_cols]

                fig, axes = joypy.joyplot(
                    df,
                    figsize=(12, 8),
                    title="Benchmark Execution Time Distribution Comparison",
                    colormap=cm.viridis_r,
                    linewidth=1,
                    legend=True,
                    overlap=0.5,
                    alpha=0.8,
                    grid=True
                )

                plt.savefig(self.output_dir / "ridgeline_plot.png", dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ Generated ridgeline_plot.png")
            else:
                print("Insufficient distribution data for ridgeline plot.")
        except ImportError:
            print("joypy package not available for ridgeline plots. Install with: pip install joypy")
        except Exception as e:
            print(f"Error generating ridgeline plot: {e}")


    def generate_summary_report(self):
        from datetime import datetime
        report_path = self.output_dir / "summary_report.txt"

        with open(report_path, 'w') as f:
            f.write("BENCHMARK ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session directory: {self.session_dir}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")

            try:
                import platform
                import psutil
                f.write("SYSTEM INFORMATION\n")
                f.write("-" * 30 + "\n")
                f.write(f"Platform: {platform.platform()}\n")
                f.write(f"Python: {platform.python_version()}\n")
                f.write(f"Processor: {platform.processor()}\n")

                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    f.write(f"CPU Frequency: {cpu_freq.current:.2f} MHz (min: {cpu_freq.min:.2f}, max: {cpu_freq.max:.2f})\n")
                f.write(f"CPU Count: {psutil.cpu_count(logical=True)} logical, {psutil.cpu_count(logical=False) or '?'} physical\n")

                memory = psutil.virtual_memory()
                f.write(f"Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available\n\n")
            except ImportError:
                f.write("System information not available (psutil package not installed)\n\n")
            except Exception as e:
                if self.debug_mode:
                    f.write(f"Error getting system info: {e}\n\n")

            if self.hyperfine_paths:
                f.write("EXECUTION TIMES\n")
                f.write("-" * 30 + "\n")

                fastest_bench = None
                fastest_time = float('inf')

                execution_data = []

                for bench_name in self.hyperfine_paths:
                    data = self._load_hyperfine_data_for_bench(bench_name)
                    clean_name = self._clean_benchmark_name(bench_name)
                    if 'json' in data:
                        results = data['json'].get('results', [])
                        if results:
                            result = results[0]
                            mean_time = result.get('mean', 0)
                            std_time = result.get('stddev', 0)
                            min_time = result.get('min', 0)
                            max_time = result.get('max', 0)

                            execution_data.append({
                                'bench_name': bench_name,
                                'clean_name': clean_name,
                                'mean': mean_time,
                                'std': std_time,
                                'min': min_time,
                                'max': max_time
                            })

                            if mean_time < fastest_time and mean_time > 0:
                                fastest_time = mean_time
                                fastest_bench = clean_name
                    del data

                execution_data.sort(key=lambda x: x['mean'])

                f.write(f"{'Benchmark':<30} {'Mean (s)':<10} {'Std Dev':<10} {'Min (s)':<10} {'Max (s)':<10} {'vs Fastest':<10}\n")
                f.write("-" * 80 + "\n")

                for data in execution_data:
                    vs_fastest = data['mean'] / fastest_time if fastest_time > 0 else 1.0
                    vs_fastest_str = f"{vs_fastest:.2f}x" if data['clean_name'] != fastest_bench else "1.00x (fastest)"

                    f.write(f"{data['clean_name'][:30]:<30} {data['mean']:.4f}     {data['std']:.4f}     "
                            f"{data['min']:.4f}     {data['max']:.4f}     {vs_fastest_str}\n")

                f.write("\n")

            resource_data = self._load_resource_data()
            if resource_data is not None and not resource_data.empty:
                f.write("RESOURCE USAGE\n")
                f.write("-" * 30 + "\n")

                f.write(f"{'Benchmark':<30} {'Memory (MB)':<12} {'CPU Time (s)':<12} {'User Time (s)':<12} {'Sys Time (s)':<12}\n")
                f.write("-" * 80 + "\n")

                for _, row in resource_data.iterrows():
                    bench_name = row['benchmark']
                    clean_name = self._clean_benchmark_name(bench_name)
                    memory_mb = row['max_rss_kb'] / 1024
                    cpu_time = row['user_time'] + row['system_time']
                    user_time = row['user_time']
                    sys_time = row['system_time']

                    f.write(f"{clean_name[:30]:<30} {memory_mb:>11.1f} {cpu_time:>11.2f} {user_time:>11.2f} {sys_time:>11.2f}\n")

                if len(resource_data) > 1:
                    f.write("-" * 80 + "\n")
                    f.write(f"{'AVERAGE':<30} {resource_data['max_rss_kb'].mean()/1024:>11.1f} "
                            f"{(resource_data['user_time'] + resource_data['system_time']).mean():>11.2f} "
                            f"{resource_data['user_time'].mean():>11.2f} {resource_data['system_time'].mean():>11.2f}\n")

                f.write("\n")
                del resource_data

            if self.perf_counter_paths:
                f.write("PERFORMANCE COUNTERS\n")
                f.write("-" * 30 + "\n")

                key_metrics = ['cycles', 'instructions', 'cache-references', 'cache-misses',
                            'branch-instructions', 'branch-misses']

                header = f"{'Benchmark':<30}"
                for metric in key_metrics:
                    header += f" {metric[:12]:<15}"
                f.write(header + " IPC      Miss Rate\n")
                f.write("-" * 120 + "\n")

                for bench_name in self.perf_counter_paths:
                    clean_name = self._clean_benchmark_name(bench_name)
                    data = self._get_perf_data(bench_name)

                    if data is not None:
                        values = {}
                        for metric in key_metrics:
                            values[metric] = self._get_perf_metric_value(data, metric)

                        ipc = values.get('instructions', 0) / values.get('cycles', 1) if values.get('cycles', 0) > 0 else 0
                        miss_rate = values.get('cache-misses', 0) / values.get('cache-references', 1) * 100 if values.get('cache-references', 0) > 0 else 0

                        line = f"{clean_name[:30]:<30}"

                        for metric in key_metrics:
                            val = values.get(metric, 0)
                            if val >= 1e9:
                                line += f" {val/1e9:>13.2f}B "
                            elif val >= 1e6:
                                line += f" {val/1e6:>13.2f}M "
                            elif val >= 1e3:
                                line += f" {val/1e3:>13.2f}K "
                            else:
                                line += f" {val:>13.0f}  "

                        line += f" {ipc:>6.2f}   {miss_rate:>6.2f}%"
                        f.write(line + "\n")

                        del data

                f.write("\n")

            if self.context_stats_paths or self.sched_latency_paths:
                f.write("SCHEDULING & CONTEXT SWITCHES\n")
                f.write("-" * 30 + "\n")

                if self.context_stats_paths:
                    f.write(f"{'Benchmark':<30} {'Context Switches':<20} {'CPU Migrations':<20} {'Page Faults':<15}\n")
                    f.write("-" * 85 + "\n")

                    for bench_name in self.context_stats_paths:
                        clean_name = self._clean_benchmark_name(bench_name)
                        data = self._get_context_stats(bench_name)

                        if data is not None:
                            cs_count = self._get_perf_metric_value(data, 'context-switches')
                            mig_count = self._get_perf_metric_value(data, 'cpu-migrations')
                            pf_count = self._get_perf_metric_value(data, 'page-faults')

                            cs_str = f"{cs_count:,.0f}" if cs_count is not None else "N/A"
                            mig_str = f"{mig_count:,.0f}" if mig_count is not None else "N/A"
                            pf_str = f"{pf_count:,.0f}" if pf_count is not None else "N/A"

                            f.write(f"{clean_name[:30]:<30} {cs_str:<20} {mig_str:<20} {pf_str:<15}\n")
                            del data

                    f.write("\n")

                if self.sched_latency_paths:
                    f.write("SCHEDULING LATENCY DETAILS\n")
                    f.write(f"{'Benchmark':<30} {'Avg Latency (ms)':<18} {'Max Latency (ms)':<18} {'Worst Task':<30}\n")
                    f.write("-" * 96 + "\n")

                    for bench_name in self.sched_latency_paths:
                        clean_name = self._clean_benchmark_name(bench_name)
                        data = self._get_sched_latency(bench_name)

                        if data is not None and not data.empty:
                            avg_latency = data['avg_delay_ms'].mean()
                            max_latency = data['max_delay_ms'].max()
                            worst_task_idx = data['max_delay_ms'].idxmax()
                            worst_task = data.loc[worst_task_idx]['task']

                            f.write(f"{clean_name[:30]:<30} {avg_latency:>17.4f} {max_latency:>17.4f} {worst_task[:30]}\n")
                            del data

                    f.write("\n")

                f.write("\nFILES GENERATED\n")
                f.write("-" * 30 + "\n")

            generated_files = list(self.output_dir.glob("*.png")) + list(self.output_dir.glob("*.txt")) + list(self.output_dir.glob("*.csv"))
            for file in sorted(generated_files):
                if file.name != "summary_report.txt":
                    file_size = file.stat().st_size / 1024
                    f.write(f"{file.name:<40} {file_size:>8.1f} KB\n")

            print(f"✓ Generated summary report: {report_path}")
            gc.collect()


    def set_plotting_style(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['figure.titlesize'] = 18
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['figure.figsize'] = [12, 8]

        plt.style.use('seaborn-v0_8-whitegrid')

        sns.set_palette("colorblind")

        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.2


    def generate_all_plots(self):
        print("Generating benchmark visualization plots...")

        import matplotlib.pyplot as plt
        import seaborn as sns
        import time

        self.set_plotting_style();

        plots = [
            ("execution times", self.plot_execution_times),
            ("performance trends", self.plot_performance_trends),
            ("resource usage", self.plot_resource_usage),
            ("performance counters", self.plot_performance_counters),
            ("context switches", self.plot_context_switches),
            ("scheduling latency", self.plot_scheduling_latency),
            ("scheduling timeline", self.plot_sched_timeline),
            ("correlation analysis", self.plot_correlation_analysis),
            ("ridgeline plot", self.plot_ridgeline)
        ]

        print(f"Generating benchmark visualization plots in: {self.output_dir}")
        print("-" * 60)

        start_time = time.time()

        for i, (name, plot_func) in enumerate(plots):
            plot_start = time.time()
            print(f"[{i+1}/{len(plots)}] Generating {name} plots...")

            try:
                plot_func()
                plot_end = time.time()
                print(f"  ✓ Completed in {plot_end - plot_start:.1f} seconds\n")
            except Exception as e:
                print(f"  ✗ Error generating {name} plots: {e}")
                if self.debug_mode:
                    import traceback
                    traceback.print_exc()

            gc.collect()

        print("Generating summary report...")
        self.generate_summary_report()

        total_time = time.time() - start_time
        print(f"\nAll plots generated in {total_time:.1f} seconds!")
        print(f"Output directory: {self.output_dir}")

        generated_files = list(self.output_dir.glob("*.png")) + list(self.output_dir.glob("*.txt")) + list(self.output_dir.glob("*.csv"))
        if generated_files:
            file_count = len(generated_files)
            print(f"\nGenerated {file_count} files:")
            for i, file in enumerate(sorted(generated_files)):
                # Print limited number of files to not overflow console
                if i < 10 or i >= len(generated_files) - 5:
                    file_size = file.stat().st_size / 1024  # KB
                    print(f"  - {file.name:<40} {file_size:>8.1f} KB")
                elif i == 10:
                    print(f"  ... ({file_count - 15} more files) ...")


    def _clean_benchmark_name(self, bench_name: str) -> str:
        if not bench_name:
            return bench_name

        cleaned = bench_name

        patterns_to_remove = [
            'bench', 'benchmark', 'test', '_test', 'test_',
            '_bench', 'bench_', '_benchmark', 'benchmark_'
        ]

        for pattern in patterns_to_remove:
            if cleaned.lower().startswith(pattern.lower()):
                cleaned = cleaned[len(pattern):]
            elif cleaned.lower().endswith(pattern.lower()):
                cleaned = cleaned[:-len(pattern)]
            cleaned = cleaned.replace(f'_{pattern}_', '_')
            cleaned = cleaned.replace(f'_{pattern}', '')
            cleaned = cleaned.replace(f'{pattern}_', '')

        cleaned = cleaned.replace('_', ' ')

        cleaned = cleaned.replace('-', ' ')

        extensions = ['.exe', '.out', '.bin', '.test']
        for ext in extensions:
            if cleaned.lower().endswith(ext):
                cleaned = cleaned[:-len(ext)]

        import re
        cleaned = re.sub(r'^.*?target/release/deps/', '', cleaned)
        cleaned = re.sub(r'^.*?target/debug/deps/', '', cleaned)

        cleaned = re.sub(r'-[a-f0-9]{16}$', '', cleaned)
        cleaned = re.sub(r'-[a-f0-9]{8,}$', '', cleaned)

        cleaned = ' '.join(cleaned.split())

        cleaned = ' '.join(word.capitalize() for word in cleaned.split())

        if not cleaned or cleaned.isspace():
            return bench_name

        return cleaned.strip()


    def _get_clean_benchmark_names(self, benchmark_list):
        name_mapping = {}
        clean_names = []

        for bench_name in benchmark_list:
            clean_name = self._clean_benchmark_name(bench_name)
            name_mapping[bench_name] = clean_name
            clean_names.append(clean_name)

        seen_names = {}
        final_clean_names = []

        for i, clean_name in enumerate(clean_names):
            if clean_name in seen_names:
                seen_names[clean_name] += 1
                final_name = f"{clean_name} ({seen_names[clean_name]})"
            else:
                seen_names[clean_name] = 1
                final_name = clean_name

            final_clean_names.append(final_name)
            name_mapping[benchmark_list[i]] = final_name

        return final_clean_names, name_mapping


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

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )

    parser.add_argument(
        '--sample-rate',
        type=float,
        default=1.0,
        help='Sample rate for large datasets (0.0-1.0, default: 1.0)'
    )

    parser.add_argument(
        '--theme',
        choices=['default', 'light', 'dark', 'colorblind'],
        default='default',
        help='Visual theme for plots (default: default)'
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
        start_time = time.time()
        print(f"Starting benchmark visualization at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        visualizer = BenchmarkVisualizer(args.session_dir, args.output)
        visualizer.debug_mode = args.debug

        import matplotlib.pyplot as plt
        plt.rcParams['figure.dpi'] = args.dpi
        plt.rcParams['savefig.format'] = args.format

        if args.theme == 'dark':
            plt.style.use('dark_background')
        elif args.theme == 'colorblind':
            import seaborn as sns
            sns.set_palette("colorblind")
        elif args.theme == 'light':
            plt.style.use('seaborn-v0_8-whitegrid')

        plot_types = [p.strip().lower() for p in args.plots.split(',')]

        if 'all' in plot_types:
            visualizer.generate_all_plots()
        else:
            plot_functions = {
                'execution': visualizer.plot_execution_times,
                'trends': visualizer.plot_performance_trends,
                'resource': visualizer.plot_resource_usage,
                'perf': visualizer.plot_performance_counters,
                'context': visualizer.plot_context_switches,
                'latency': visualizer.plot_scheduling_latency,
                'timeline': visualizer.plot_sched_timeline,
                'correlation': visualizer.plot_correlation_analysis,
                'ridgeline': visualizer.plot_ridgeline
            }

            plot_count = sum(1 for p in plot_types if p in plot_functions)
            print(f"Generating {plot_count} plot types...")

            for i, plot_type in enumerate(plot_types):
                if plot_type in plot_functions:
                    print(f"[{i+1}/{plot_count}] Generating {plot_type} plots...")
                    try:
                        plot_start = time.time()
                        plot_functions[plot_type]()
                        print(f"  ✓ Completed in {time.time() - plot_start:.1f} seconds")
                    except Exception as e:
                        print(f"  ✗ Error generating {plot_type} plots: {e}")
                        if args.debug:
                            import traceback
                            traceback.print_exc()
                    gc.collect()

            visualizer.generate_summary_report()
            gc.collect()

        total_time = time.time() - start_time
        print(f"\nVisualization completed in {total_time:.1f} seconds")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
