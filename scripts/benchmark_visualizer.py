#!/usr/bin/env python3

import argparse
import json
import re
import sys
import gc
from pathlib import Path
from typing import Optional, Dict, List
import warnings

warnings.filterwarnings('ignore')

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

        import random
        if sample_rate < 1.0:
            sample_size = max(1000, int(len(rows) * sample_rate))
            rows = random.sample(rows, min(sample_size, len(rows)))

        import pandas as pd
        return pd.DataFrame(rows)


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

        fig = plt.figure(figsize=(18, 12))

        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(clean_names, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('Benchmark')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Mean Execution Times with Standard Deviation')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        if use_log_scale:
            ax1.set_yscale('log')
            ax1.set_ylabel('Execution Time (seconds) - Log Scale')

        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            offset = std if not use_log_scale else height * 0.1
            ax1.text(bar.get_x() + bar.get_width()/2., height + offset,
                    f'{mean:.3f}s±{std:.3f}s', ha='center', va='bottom', fontsize=8)

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
                for patch in bp['boxes']:
                    patch.set_facecolor('lightgreen')
                    patch.set_alpha(0.7)
                ax2.set_xlabel('Benchmark')
                ax2.set_ylabel('Execution Time (seconds)')
                ax2.set_title('Execution Time Distributions')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)

                if use_log_scale:
                    ax2.set_yscale('log')
                    ax2.set_ylabel('Execution Time (seconds) - Log Scale')
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
            ax3.errorbar(x_pos, means, yerr=[np.array(means) - np.array(mins),
                                            np.array(maxs) - np.array(means)],
                        fmt='o', capsize=5, capthick=2, markersize=8, alpha=0.8)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(clean_names, rotation=45)
            ax3.set_xlabel('Benchmark')
            ax3.set_ylabel('Execution Time (seconds)')
            ax3.set_title('Min/Max Range with Mean')
            ax3.grid(True, alpha=0.3)

            if use_log_scale:
                ax3.set_yscale('log')
                ax3.set_ylabel('Execution Time (seconds) - Log Scale')

            for i, (clean_name, mean_val, min_val, max_val) in enumerate(zip(clean_names, means, mins, maxs)):
                range_val = max_val - min_val
                offset = mean_val * 0.1 if use_log_scale else 0
                ax3.annotate(f'Range: {range_val:.4f}s',
                            xy=(i, mean_val + offset), xytext=(5, 10),
                            textcoords='offset points', fontsize=8, alpha=0.7)
        else:
            ax3.text(0.5, 0.5, 'No min/max data available',
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('Min/Max Range (No Data)')

        ax4 = fig.add_subplot(gs[1, 1])
        if means:
            fastest_time = min(means)
            relative_performance = [mean / fastest_time for mean in means]

            colors = ['green' if rp == 1.0 else 'orange' if rp < 1.5 else 'red' for rp in relative_performance]
            bars = ax4.bar(clean_names, relative_performance, color=colors, alpha=0.7)

            ax4.set_xlabel('Benchmark')
            ax4.set_ylabel('Relative Performance (vs fastest)')
            ax4.set_title('Relative Performance Comparison')
            ax4.tick_params(axis='x', rotation=45)
            ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline (fastest)')
            ax4.grid(True, alpha=0.3)
            ax4.legend()

            for bar, rp, mean in zip(bars, relative_performance, means):
                height = bar.get_height()
                if rp == 1.0:
                    label = f'Fastest\n{mean:.3f}s'
                else:
                    slower_pct = (rp - 1.0) * 100
                    label = f'+{slower_pct:.1f}%\n{mean:.3f}s'
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        label, ha='center', va='bottom', fontsize=8)

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

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        if 'datetime' in resource_data.columns:
            resource_data_sorted = resource_data.sort_values('datetime').reset_index(drop=True)

            x = np.arange(len(resource_data_sorted))
            y = resource_data_sorted['elapsed_time']

            if len(x) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                trend_line = slope * x + intercept

                ax1.scatter(x, y, alpha=0.7, s=60)
                ax1.plot(x, trend_line, 'r--', alpha=0.8,
                        label=f'Trend: slope={slope:.4f}, R²={r_value**2:.3f}')
                ax1.set_xlabel('Benchmark Sequence')
                ax1.set_ylabel('Execution Time (seconds)')
                ax1.set_title('Performance Trend Analysis')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            if len(resource_data_sorted) > 1:
                rolling_mean = pd.Series(y).rolling(window=min(3, len(y))).mean()
                rolling_std = pd.Series(y).rolling(window=min(3, len(y))).std()

                ax2.plot(x, y, 'bo-', alpha=0.7, label='Actual')
                ax2.plot(x, rolling_mean, 'r-', linewidth=2, label='Rolling Mean')
                ax2.fill_between(x,
                            rolling_mean - rolling_std,
                            rolling_mean + rolling_std,
                            alpha=0.3, label='±1 Std Dev')
                ax2.set_xlabel('Benchmark Sequence')
                ax2.set_ylabel('Execution Time (seconds)')
                ax2.set_title('Performance Stability Analysis')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            cpu_time = resource_data_sorted['user_time'] + resource_data_sorted['system_time']
            memory_mb = resource_data_sorted['max_rss_kb'] / 1024

            scatter = ax3.scatter(cpu_time, memory_mb,
                                c=resource_data_sorted['elapsed_time'],
                                s=80, alpha=0.7, cmap='viridis')
            ax3.set_xlabel('CPU Time (seconds)')
            ax3.set_ylabel('Peak Memory (MB)')
            ax3.set_title('Resource Usage Pattern')
            plt.colorbar(scatter, ax=ax3, label='Execution Time (s)')
            ax3.grid(True, alpha=0.3)

            ax4.hist(resource_data_sorted['elapsed_time'], bins=min(10, len(resource_data_sorted)),
                    alpha=0.7, edgecolor='black')
            ax4.axvline(resource_data_sorted['elapsed_time'].mean(),
                    color='red', linestyle='--', label='Mean')
            ax4.axvline(resource_data_sorted['elapsed_time'].median(),
                    color='green', linestyle='--', label='Median')
            ax4.set_xlabel('Execution Time (seconds)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Performance Distribution')
            ax4.legend()
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

        ax1.bar(x - width/2, user_time, width, label='User Time', alpha=0.7)
        ax1.bar(x + width/2, system_time, width, label='System Time', alpha=0.7)
        ax1.set_xlabel('Benchmark')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('CPU Time Breakdown')
        ax1.set_xticks(x)
        ax1.set_xticklabels(clean_benchmarks, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        all_times = np.concatenate([user_time, system_time])
        if len(all_times) > 0 and np.max(all_times) > 0 and np.min(all_times[all_times > 0]) > 0:
            ratio = np.max(all_times) / np.min(all_times[all_times > 0])
            if ratio > 10:
                ax1.set_yscale('log')
                ax1.set_ylabel('Time (seconds) - Log Scale')

        memory_mb = resource_data['max_rss_kb'].values / 1024
        bars = ax2.bar(clean_benchmarks, memory_mb, alpha=0.7)
        ax2.set_xlabel('Benchmark')
        ax2.set_ylabel('Peak Memory Usage (MB)')
        ax2.set_title('Peak Memory Usage')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        if len(memory_mb) > 0 and np.max(memory_mb) > 0 and np.min(memory_mb[memory_mb > 0]) > 0:
            ratio = np.max(memory_mb) / np.min(memory_mb[memory_mb > 0])
            if ratio > 10:
                ax2.set_yscale('log')
                ax2.set_ylabel('Peak Memory Usage (MB) - Log Scale')

        for bar, mem in zip(bars, memory_mb):
            height = bar.get_height()
            offset = height * 0.1 if ax2.get_yscale() == 'log' else 0
            ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                    f'{mem:.1f}MB', ha='center', va='bottom', fontsize=8)

        total_time = resource_data['elapsed_time'].values
        cpu_time = user_time + system_time
        cpu_efficiency = (cpu_time / total_time) * 100

        ax3.bar(clean_benchmarks, cpu_efficiency, alpha=0.7)
        ax3.set_xlabel('Benchmark')
        ax3.set_ylabel('CPU Efficiency (%)')
        ax3.set_title('CPU Utilization Efficiency')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)

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

            plot_df = pd.melt(
                metrics_df,
                id_vars=['Benchmark'],
                value_vars=['Time Score', 'Memory Score', 'CPU Score'],
                var_name='Metric',
                value_name='Score'
            )

            sns.heatmap(
                metrics_df.set_index('Benchmark')[['Time Score', 'Memory Score', 'CPU Score']],
                annot=True,
                cmap='RdYlGn',
                linewidths=1,
                ax=ax4,
                vmin=0,
                vmax=100,
                fmt='.1f'
            )

            ax4.set_title('Performance Scores')
            ax4.set_ylabel('Benchmark')
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

                bars = axes[i].bar(clean_names, values, alpha=0.7)
                axes[i].set_xlabel('Benchmark')
                axes[i].set_ylabel('Count')
                axes[i].set_title(f'{metric.replace("-", " ").title()}')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)

                if len(values) > 0 and max(values) > 0 and min([v for v in values if v > 0]) > 0:
                    positive_values = [v for v in values if v > 0]
                    if positive_values:
                        ratio = max(positive_values) / min(positive_values)
                        if ratio > 100:
                            axes[i].set_yscale('log')
                            axes[i].set_ylabel('Count - Log Scale')

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

                        offset = height * 0.1 if axes[i].get_yscale() == 'log' else 0
                        axes[i].text(bar.get_x() + bar.get_width()/2., height + offset,
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
        plt.close(fig)
        print("✓ Generated performance_counters.png")
        gc.collect()


    def plot_context_switches(self):
        if not self.context_stats_paths:
            print("No context switch data available for plotting.")
            return

        import matplotlib.pyplot as plt
        import numpy as np

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

        bars1 = ax1.bar(clean_names, context_switches, alpha=0.7)
        ax1.set_xlabel('Benchmark')
        ax1.set_ylabel('Context Switches')
        ax1.set_title('Context Switches per Benchmark')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        if use_log_cs:
            ax1.set_yscale('log')
            ax1.set_ylabel('Context Switches - Log Scale')

        use_log_mig = False
        if cpu_migrations and max(cpu_migrations) > 0 and min([m for m in cpu_migrations if m > 0]) > 0:
            positive_mig = [m for m in cpu_migrations if m > 0]
            if positive_mig:
                ratio = max(positive_mig) / min(positive_mig)
                use_log_mig = ratio > 100

        bars2 = ax2.bar(clean_names, cpu_migrations, alpha=0.7, color='orange')
        ax2.set_xlabel('Benchmark')
        ax2.set_ylabel('CPU Migrations')
        ax2.set_title('CPU Migrations per Benchmark')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        if use_log_mig:
            ax2.set_yscale('log')
            ax2.set_ylabel('CPU Migrations - Log Scale')

        use_log_pf = False
        if page_faults and max(page_faults) > 0 and min([pf for pf in page_faults if pf > 0]) > 0:
            positive_pf = [pf for pf in page_faults if pf > 0]
            if positive_pf:
                ratio = max(positive_pf) / min(positive_pf)
                use_log_pf = ratio > 100

        bars3 = ax3.bar(clean_names, page_faults, alpha=0.7, color='red')
        ax3.set_xlabel('Benchmark')
        ax3.set_ylabel('Page Faults')
        ax3.set_title('Page Faults per Benchmark')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

        if use_log_pf:
            ax3.set_yscale('log')
            ax3.set_ylabel('Page Faults - Log Scale')

        if any(tc > 0 for tc in task_clock):
            cs_per_sec = [cs / (tc / 1000) if tc > 0 else 0
                        for cs, tc in zip(context_switches, task_clock)]

            use_log_rate = False
            if cs_per_sec and max(cs_per_sec) > 0 and min([rate for rate in cs_per_sec if rate > 0]) > 0:
                positive_rates = [rate for rate in cs_per_sec if rate > 0]
                if positive_rates:
                    ratio = max(positive_rates) / min(positive_rates)
                    use_log_rate = ratio > 100

            ax4.bar(clean_names, cs_per_sec, alpha=0.7, color='green')
            ax4.set_xlabel('Benchmark')
            ax4.set_ylabel('Context Switches/sec')
            ax4.set_title('Context Switch Rate')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)

            if use_log_rate:
                ax4.set_yscale('log')
                ax4.set_ylabel('Context Switches/sec - Log Scale')
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
                sizes = 20 + (sizes - sizes.min()) / (sizes.max() - sizes.min() + 0.1) * 100

                scatter = ax.scatter(data['avg_delay_ms'], data['max_delay_ms'],
                                s=sizes, alpha=0.6, c=data['switches'],
                                cmap='viridis')
                ax.set_xlabel('Average Delay (ms)')
                ax.set_ylabel('Maximum Delay (ms)')
                ax.set_title(f'{clean_name} - Scheduling Latency')
                ax.grid(True, alpha=0.3)

                avg_delays = data['avg_delay_ms'].values
                max_delays = data['max_delay_ms'].values
                all_delays = np.concatenate([avg_delays[avg_delays > 0], max_delays[max_delays > 0]])

                if len(all_delays) > 0:
                    ratio = np.max(all_delays) / np.min(all_delays)
                    if ratio > 100:
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                        ax.set_xlabel('Average Delay (ms) - Log Scale')
                        ax.set_ylabel('Maximum Delay (ms) - Log Scale')

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

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

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

        bars1 = ax1.bar(clean_names, avg_latencies, alpha=0.7)
        ax1.set_xlabel('Benchmark')
        ax1.set_ylabel('Average Scheduling Latency (ms)')
        ax1.set_title('Average Scheduling Latency Comparison')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        if use_log_avg:
            ax1.set_yscale('log')
            ax1.set_ylabel('Average Scheduling Latency (ms) - Log Scale')

        for bar, val in zip(bars1, avg_latencies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}ms', ha='center', va='bottom', fontsize=8)

        use_log_worst = False
        if worst_task_latencies and max(worst_task_latencies) > 0 and min([l for l in worst_task_latencies if l > 0]) > 0:
            positive_worst = [l for l in worst_task_latencies if l > 0]
            if positive_worst:
                ratio = max(positive_worst) / min(positive_worst)
                use_log_worst = ratio > 10

        bars2 = ax2.bar(clean_names, worst_task_latencies, alpha=0.7, color='red')
        ax2.set_xlabel('Benchmark')
        ax2.set_ylabel('Worst Task Latency (ms)')
        ax2.set_title('Worst Task Scheduling Latency')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        if use_log_worst:
            ax2.set_yscale('log')
            ax2.set_ylabel('Worst Task Latency (ms) - Log Scale')

        for i, (bar, val, task) in enumerate(zip(bars2, worst_task_latencies, worst_tasks)):
            height = bar.get_height()
            task_short = task[:10] + '...' if len(task) > 10 else task
            offset = height * 0.1 if use_log_worst else 0
            ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{val:.3f}ms\n{task_short}', ha='center', va='bottom', fontsize=8)

        use_log_switches = False
        if total_switches and max(total_switches) > 0 and min([s for s in total_switches if s > 0]) > 0:
            positive_switches = [s for s in total_switches if s > 0]
            if positive_switches:
                ratio = max(positive_switches) / min(positive_switches)
                use_log_switches = ratio > 100

        bars3 = ax3.bar(clean_names, total_switches, alpha=0.7, color='green')
        ax3.set_xlabel('Benchmark')
        ax3.set_ylabel('Total Context Switches')
        ax3.set_title('Total Context Switches')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

        if use_log_switches:
            ax3.set_yscale('log')
            ax3.set_ylabel('Total Context Switches - Log Scale')

        ax4.scatter(total_switches, worst_task_latencies, alpha=0.7, s=60)
        ax4.set_xlabel('Total Context Switches')
        ax4.set_ylabel('Worst Task Latency (ms)')
        ax4.set_title('Latency vs Context Switches')
        ax4.grid(True, alpha=0.3)

        if use_log_switches:
            ax4.set_xscale('log')
            ax4.set_xlabel('Total Context Switches - Log Scale')
        if use_log_worst:
            ax4.set_yscale('log')
            ax4.set_ylabel('Worst Task Latency (ms) - Log Scale')

        for i, clean_name in enumerate(clean_names):
            ax4.annotate(clean_name[:16],
                        (total_switches[i], worst_task_latencies[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)

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

        fig, ax = plt.subplots(figsize=(12, 8))

        task_labels = [f"{row['task'][:15]}...\n({row['clean_benchmark']})" if len(row['task']) > 15
                    else f"{row['task']}\n({row['clean_benchmark']})"
                    for _, row in top_latency_tasks.iterrows()]

        x = np.arange(len(task_labels))
        width = 0.35

        avg_delays = top_latency_tasks['avg_delay'].values
        max_delays = top_latency_tasks['max_delay'].values

        all_delays = np.concatenate([avg_delays[avg_delays > 0], max_delays[max_delays > 0]])
        use_log_task = False
        if len(all_delays) > 0:
            ratio = np.max(all_delays) / np.min(all_delays)
            use_log_task = ratio > 10

        ax.bar(x - width/2, avg_delays, width, label='Avg Delay (ms)', alpha=0.7)
        ax.bar(x + width/2, max_delays, width, label='Max Delay (ms)', alpha=0.7, color='red')

        ax.set_ylabel('Delay (ms)')
        ax.set_title('Top Tasks by Scheduling Latency')
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if use_log_task:
            ax.set_yscale('log')
            ax.set_ylabel('Delay (ms) - Log Scale')

        for i, switches in enumerate(top_latency_tasks['switches']):
            ax.text(i, 0.1 if not use_log_task else min(all_delays) * 0.1, f"{switches} sw",
                    ha='center', va='bottom', fontsize=8, rotation=90, alpha=0.7)

        plt.tight_layout()
        plt.savefig(self.output_dir / "task_latency_distribution.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Generated task_latency_distribution.png")

        del latency_df, top_latency_tasks
        gc.collect()


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
                    plt.close(fig)
                    print("✓ Generated correlation_analysis.png")

                    del correlation_matrix
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


    def plot_sched_timeline(self):
        if not self.sched_timeline_paths:
            print("No scheduling timeline data available for plotting.")
            return

        import matplotlib.pyplot as plt
        import numpy as np

        sample_rate = 1.0
        benchmark_names = list(self.sched_timeline_paths.keys())
        clean_names, name_mapping = self._get_clean_benchmark_names(benchmark_names)

        for i, bench_name in enumerate(self.sched_timeline_paths):
            clean_name = clean_names[i]

            sample_data = self._get_sched_timeline(bench_name, sample_rate=0.01)
            if sample_data is None or sample_data.empty:
                continue

            estimated_total = len(sample_data) * 100
            if estimated_total > 50000:
                sample_rate = min(1.0, 50000 / estimated_total)
                print(f"Large timeline dataset detected for {clean_name}, using {sample_rate:.1%} sample")
            del sample_data

            timeline_data = self._get_sched_timeline(bench_name, sample_rate=sample_rate)
            if timeline_data is None or timeline_data.empty:
                continue

            print(f"Generating scheduling timeline visualization for {clean_name}...")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})

            base_time = timeline_data['timestamp'].min()

            relative_ms = (timeline_data['timestamp'] - base_time) * 1000

            cpus = sorted(timeline_data['cpu'].unique())

            task_counts = timeline_data['task_name'].value_counts().head(10)
            top_tasks = task_counts.index.tolist()

            colors = plt.cm.tab10(np.linspace(0, 1, len(top_tasks)))
            task_colors = dict(zip(top_tasks, colors))

            for cpu in cpus:
                cpu_data = timeline_data[timeline_data['cpu'] == cpu]

                if not cpu_data.empty:
                    for task in top_tasks:
                        task_data = cpu_data[cpu_data['task_name'] == task]
                        if not task_data.empty:
                            ax1.scatter(
                                (task_data['timestamp'] - base_time) * 1000,
                                [cpu] * len(task_data),
                                color=task_colors[task],
                                s=30,
                                alpha=0.7,
                                label=f"{task}" if cpu == cpus[0] else ""
                            )
                            del task_data

                for _, row in cpu_data.iterrows():
                    if row['run_time_ms'] > 0.1:  # Only show significant run times
                        start_ms = (row['timestamp'] - base_time) * 1000
                        end_ms = start_ms + row['run_time_ms']
                        ax1.hlines(
                            y=cpu,
                            xmin=start_ms,
                            xmax=end_ms,
                            colors='gray',
                            linewidth=2,
                            alpha=0.3
                        )

                del cpu_data
                gc.collect()

            delays = timeline_data['sched_delay_ms'].values
            if delays.any():
                non_zero_delays = delays[delays > 0]
                if len(non_zero_delays) > 0:
                    ax2.hist(non_zero_delays, bins=30, alpha=0.7)
                    ax2.set_xlabel('Scheduling Delay (ms)')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('Scheduling Delay Distribution')

                    mean_delay = non_zero_delays.mean()
                    median_delay = np.median(non_zero_delays)
                    ax2.axvline(mean_delay, color='red', linestyle='--', alpha=0.8,
                            label=f'Mean: {mean_delay:.3f} ms')
                    ax2.axvline(median_delay, color='green', linestyle='--', alpha=0.8,
                            label=f'Median: {median_delay:.3f} ms')
                    ax2.legend()
                else:
                    ax2.text(0.5, 0.5, 'No non-zero scheduling delays found',
                            transform=ax2.transAxes, ha='center', va='center')
            else:
                ax2.text(0.5, 0.5, 'No scheduling delay data found',
                        transform=ax2.transAxes, ha='center', va='center')

            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('CPU')
            ax1.set_title(f'Scheduling Timeline - {clean_name}')

            ax1.set_yticks(cpus)
            ax1.set_yticklabels([f'CPU {cpu}' for cpu in cpus])

            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax1.legend(by_label.values(), by_label.keys(),
                    loc='upper right', fontsize='small')

            ax1.grid(True, axis='x', alpha=0.3)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            safe_name = clean_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            plt.savefig(self.output_dir / f"{safe_name}_sched_timeline.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Generated {safe_name}_sched_timeline.png")

            self._plot_task_timeline(bench_name, clean_name, timeline_data, top_tasks, task_colors)

            del timeline_data
            gc.collect()


    def _plot_task_timeline(self, bench_name, clean_name, timeline_data, top_tasks, task_colors):
        if timeline_data is None or timeline_data.empty:
            return

        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(15, 8))

        base_time = timeline_data['timestamp'].min()

        for i, task in enumerate(top_tasks):
            task_data = timeline_data[timeline_data['task_name'] == task]

            if not task_data.empty:
                x = (task_data['timestamp'].values - base_time) * 1000
                y = [i] * len(task_data)
                size = task_data['run_time_ms'].values * 10 + 10
                colors = plt.cm.tab20(task_data['cpu'].values % 20 / 20)

                scatter = ax.scatter(x, y, s=size, c=colors, alpha=0.7)

                if len(task_data) > 1:
                    sorted_data = task_data.sort_values('timestamp')
                    ax.plot((sorted_data['timestamp'].values - base_time) * 1000,
                        [i] * len(sorted_data),
                        'k-', alpha=0.2)
                    del sorted_data

                del task_data, x, y, size, colors

        ax.set_yticks(range(len(top_tasks)))
        ax.set_yticklabels([t[:30] + ('...' if len(t) > 30 else '') for t in top_tasks])

        ax.set_xlabel('Time from start (ms)')
        ax.set_title(f'Task Activity Timeline - {clean_name}')
        ax.grid(True, axis='x', alpha=0.3)

        sm = plt.cm.ScalarMappable(cmap=plt.cm.tab20,
                                norm=plt.Normalize(0, 19))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('CPU')

        cpu_ticks = sorted(timeline_data['cpu'].unique())
        cbar.set_ticks([tick % 20 / 20 for tick in cpu_ticks])
        cbar.set_ticklabels([f'CPU {tick}' for tick in cpu_ticks])

        plt.tight_layout()
        safe_name = clean_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        plt.savefig(self.output_dir / f"{safe_name}_task_timeline.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Generated {safe_name}_task_timeline.png")
        gc.collect()


    def generate_summary_report(self):
        from datetime import datetime
        report_path = self.output_dir / "summary_report.txt"

        with open(report_path, 'w') as f:
            f.write("BENCHMARK ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session directory: {self.session_dir}\n\n")

            if self.hyperfine_paths:
                f.write("EXECUTION TIMES\n")
                f.write("-" * 20 + "\n")
                for bench_name in self.hyperfine_paths:
                    data = self._load_hyperfine_data_for_bench(bench_name)
                    if 'json' in data:
                        results = data['json'].get('results', [])
                        if results:
                            result = results[0]
                            mean_time = result.get('mean', 0)
                            std_time = result.get('stddev', 0)
                            f.write(f"{bench_name}: {mean_time:.4f}s ± {std_time:.4f}s\n")
                    del data
                f.write("\n")

            resource_data = self._load_resource_data()
            if resource_data is not None and not resource_data.empty:
                f.write("RESOURCE USAGE\n")
                f.write("-" * 20 + "\n")
                for _, row in resource_data.iterrows():
                    bench_name = row['benchmark']
                    memory_mb = row['max_rss_kb'] / 1024
                    cpu_time = row['user_time'] + row['system_time']
                    f.write(f"{bench_name}:\n")
                    f.write(f"  Peak Memory: {memory_mb:.1f} MB\n")
                    f.write(f"  CPU Time: {cpu_time:.3f}s (User: {row['user_time']:.3f}s, System: {row['system_time']:.3f}s)\n")
                f.write("\n")
                del resource_data

            if self.perf_counter_paths:
                f.write("PERFORMANCE COUNTERS\n")
                f.write("-" * 25 + "\n")
                for bench_name in self.perf_counter_paths:
                    data = self._get_perf_data(bench_name)
                    if data is not None:
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
                        del data
                        gc.collect()

            if self.sched_latency_paths:
                f.write("SCHEDULING LATENCY\n")
                f.write("-" * 20 + "\n")
                for bench_name in self.sched_latency_paths:
                    data = self._get_sched_latency(bench_name)
                    if data is not None and not data.empty:
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

                        del data
                        gc.collect()
                f.write("\n")

                if self.context_stats_paths:
                    f.write("CONTEXT SWITCHES & MIGRATIONS\n")
                    f.write("-" * 30 + "\n")
                    for bench_name in self.context_stats_paths:
                        data = self._get_context_stats(bench_name)
                        if data is not None:
                            cs_count = self._get_perf_metric_value(data, 'context-switches')
                            mig_count = self._get_perf_metric_value(data, 'cpu-migrations')

                            f.write(f"{bench_name}:")
                            if cs_count is not None:
                                f.write(f" {cs_count:.0f} context switches")
                            if mig_count is not None:
                                f.write(f", {mig_count:.0f} CPU migrations")
                            f.write("\n")

                            del data
                    f.write("\n")

                if self.sched_timeline_paths:
                    f.write("\nSCHEDULING TIMELINE\n")
                    f.write("-" * 20 + "\n")
                    for bench_name in self.sched_timeline_paths:
                        timeline_data = self._get_sched_timeline(bench_name, sample_rate=0.2)
                        if timeline_data is not None and not timeline_data.empty:
                            total_events = len(timeline_data) * (1/0.2)  # Estimate total from sample
                            unique_tasks = timeline_data['task_name'].nunique()
                            unique_cpus = timeline_data['cpu'].nunique()
                            avg_sched_delay = timeline_data['sched_delay_ms'].mean() if 'sched_delay_ms' in timeline_data.columns else 0

                            f.write(f"{bench_name}:\n")
                            f.write(f"  Events Recorded: {int(total_events):,}\n")
                            f.write(f"  Unique Tasks: {unique_tasks}\n")
                            f.write(f"  CPUs Utilized: {unique_cpus}\n")
                            f.write(f"  Avg Scheduling Delay: {avg_sched_delay:.4f} ms\n")

                            del timeline_data
                            gc.collect()
                    f.write("\n")

            print(f"✓ Generated summary report: {report_path}")
            gc.collect()


    def generate_all_plots(self):
        print("Generating benchmark visualization plots...")

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        self.plot_execution_times()
        gc.collect()

        self.plot_performance_trends()
        gc.collect()

        self.plot_resource_usage()
        gc.collect()

        self.plot_performance_counters()
        gc.collect()

        self.plot_context_switches()
        gc.collect()

        self.plot_scheduling_latency()
        gc.collect()

        self.plot_sched_timeline()
        gc.collect()

        self.plot_correlation_analysis()
        gc.collect()

        self.generate_summary_report()
        gc.collect()

        print("\nAll plots generated successfully!")
        print(f"Output directory: {self.output_dir}")

        generated_files = list(self.output_dir.glob("*.png")) + list(self.output_dir.glob("*.txt"))
        if generated_files:
            print("\nGenerated files:")
            for file in sorted(generated_files):
                print(f"  - {file.name}")


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
        visualizer.debug_mode = args.debug

        import matplotlib.pyplot as plt
        plt.rcParams['figure.dpi'] = args.dpi
        plt.rcParams['savefig.format'] = args.format

        plot_types = [p.strip().lower() for p in args.plots.split(',')]

        if 'all' in plot_types:
            visualizer.generate_all_plots()
        else:
            if 'execution' in plot_types:
                visualizer.plot_execution_times()
                gc.collect()
            if 'resource' in plot_types:
                visualizer.plot_resource_usage()
                gc.collect()
            if 'perf' in plot_types:
                visualizer.plot_performance_counters()
                gc.collect()
            if 'context' in plot_types:
                visualizer.plot_context_switches()
                gc.collect()
            if 'latency' in plot_types:
                visualizer.plot_scheduling_latency()
                gc.collect()
            if 'timeline' in plot_types:
                visualizer.plot_sched_timeline()
                gc.collect()
            if 'correlation' in plot_types:
                visualizer.plot_correlation_analysis()
                gc.collect()
            if 'trends' in plot_types:
                visualizer.plot_performance_trends()
                gc.collect()

            visualizer.generate_summary_report()
            gc.collect()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
