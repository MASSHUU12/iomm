#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Optional, Callable, Union, Tuple, Any

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark visualization."""
    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))
    dpi: int = 100
    fig_size: Tuple[int, int] = (10, 6)

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True)


class PlotBuilder:
    """Class to handle plot creation with consistent styling."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def save_figure(self, fig: plt.Figure, filename: str) -> None:
        """Tighten layout, save figure, and log path."""
        path = self.config.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=self.config.dpi)
        print(f"[+] Saved figure to {path}")

    def create_bar_plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        yerr: Optional[str] = None,
        title: str = "",
        ylabel: str = "",
        log_y: bool = False,
        color: str = "skyblue",
        fmt_y_major: bool = False,
        fmt_y_minor: bool = False,
        ax: Optional[plt.Axes] = None,
        fig_kwargs: Optional[dict] = None,
    ) -> plt.Axes:
        """Create a bar plot with consistent styling."""
        fig_kwargs = fig_kwargs or {"figsize": self.config.fig_size}
        if ax is None:
            fig, ax = plt.subplots(**fig_kwargs)
        else:
            fig = ax.figure

        ax.bar(
            df[x],
            df[y],
            yerr=(df[yerr] if yerr else None),
            capsize=5,
            color=color,
        )

        if log_y:
            ax.set_yscale("log")
        if fmt_y_major:
            ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
        if fmt_y_minor:
            ax.yaxis.set_minor_formatter(mtick.NullFormatter())

        ax.set_title(title)
        ax.set_xlabel(x.capitalize())
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45, labelrotation=45)
        ax.grid(which="major", axis="y", linestyle="--", linewidth=0.5)
        return ax

    def create_violin_plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        log_scale: bool = False,
        symlog: bool = False,
        linthresh: float = 50.0,
        title: str = "",
        ylabel: str = "",
        ax: Optional[plt.Axes] = None,
        fig_kwargs: Optional[dict] = None,
    ) -> plt.Axes:
        """Create a violin plot with consistent styling."""
        fig_kwargs = fig_kwargs or {"figsize": self.config.fig_size}
        if ax is None:
            fig, ax = plt.subplots(**fig_kwargs)
        else:
            fig = ax.figure

        sns.violinplot(
            x=x,
            y=y,
            data=df,
            inner="quartile",
            density_norm="width",
            cut=0,
            ax=ax,
        )

        if log_scale:
            ax.set_yscale("log")
        elif symlog:
            ax.set_yscale("symlog", linthresh=linthresh)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        return ax


class BenchmarkDataLoader:
    """Class to handle loading and processing benchmark data."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def load_csvs(
        self,
        pattern: str,
        *,
        rename_columns: Dict[str, str],
        assign_cols: Optional[Dict[str, Union[str, Callable[[pd.DataFrame], pd.Series]]]] = None,
        filter_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Load multiple CSVs matching pattern and process them."""
        dfs: List[pd.DataFrame] = []
        for path in sorted(self.config.output_dir.glob(pattern)):
            df = pd.read_csv(path)
            if assign_cols:
                df = df.assign(**{
                    new_col: (func(df) if callable(func) else df[func])
                    for new_col, func in assign_cols.items()
                })
            df = df.rename(columns=rename_columns)
            if filter_fn:
                df = filter_fn(df)
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def load_hyperfine_csvs(self) -> pd.DataFrame:
        """Load Hyperfine CSV results and process them."""
        files = sorted(self.config.output_dir.glob("*_time.csv"))
        records = []
        for path in files:
            name = path.stem.replace("_time", "")
            df = pd.read_csv(path)
            df = df.assign(
                benchmark=name,
                mean_ms=df["mean"] * 1_000,
                stddev_ms=df["stddev"] * 1_000,
            )
            records.append(df[["benchmark", "mean_ms", "stddev_ms"]])
        if not records:
            print("No Hyperfine CSV results found.")
            return pd.DataFrame()
        return pd.concat(records, ignore_index=True)

    def load_hyperfine_json(self) -> pd.DataFrame:
        """Load Hyperfine JSON results with per-run data."""
        files = sorted(self.config.output_dir.glob("*_time.json"))
        runs = []
        for path in files:
            name = path.stem.replace("_time", "")
            data = json.loads(path.read_text())
            for result in data.get("results", []):
                for t in result.get("times", []):
                    runs.append({"benchmark": name, "runtime_ms": t * 1_000})

        if not runs:
            print("No per-run timing data found in Hyperfine JSON.")
            return pd.DataFrame()

        return pd.DataFrame(runs)

    def load_gnutime(self) -> pd.DataFrame:
        """Load GNU time results."""
        path = self.config.output_dir / "cpu_mem.csv"
        if not path.exists():
            print("No GNU time results found.")
            return pd.DataFrame()

        df = pd.read_csv(
            path,
            header=None,
            names=["benchmark", "elapsed_s", "user_s", "sys_s", "max_rss_kb"]
        )
        df["max_rss_mb"] = df["max_rss_kb"] / 1024
        return df

    def load_perf(self) -> pd.DataFrame:
        """Load perf stat results."""
        files = sorted(self.config.output_dir.glob("*_perf.csv"))
        frames: List[pd.DataFrame] = []

        for path in files:
            name = path.stem.replace("_perf", "")
            rows = []
            with path.open(newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or row[0].startswith("#") or len(row) < 3:
                        continue
                    try:
                        val = float(row[0])
                    except ValueError:
                        continue
                    event = row[2]
                    rows.append({"benchmark": name, "event": event, "value": val})
            if rows:
                frames.append(pd.DataFrame(rows))

        if not frames:
            print("No perf stat results found.")
            return pd.DataFrame()

        perf = pd.concat(frames, ignore_index=True)
        return perf.pivot(index="benchmark", columns="event", values="value")


class BenchmarkVisualizer:
    """Main class for benchmark visualization."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.plot_builder = PlotBuilder(config)
        self.data_loader = BenchmarkDataLoader(config)

    def visualize_hyperfine(self) -> pd.DataFrame:
        """Create visualizations from Hyperfine data."""
        hf = self.data_loader.load_hyperfine_csvs()
        if hf.empty:
            return hf

        ax = self.plot_builder.create_bar_plot(
            hf,
            x="benchmark",
            y="mean_ms",
            yerr="stddev_ms",
            title="Hyperfine: Mean runtimes with stddev (ms) – Log Scale",
            ylabel="Mean runtime (ms)",
            log_y=True,
            fmt_y_major=True,
            fmt_y_minor=True,
        )
        self.plot_builder.save_figure(ax.figure, "hyperfine_means.png")
        return hf

    def visualize_hyperfine_speedup(self, hf: pd.DataFrame) -> pd.DataFrame:
        """Create speedup visualizations from Hyperfine data."""
        if hf.empty:
            print("No data for hyperfine speedup.")
            return hf

        baseline = hf["mean_ms"].min()
        hf2 = hf.assign(speedup=lambda d: baseline / d["mean_ms"])

        ax = self.plot_builder.create_bar_plot(
            hf2,
            x="benchmark",
            y="speedup",
            title="Hyperfine: Relative Speedup over Fastest (×)",
            ylabel="Speedup factor",
        )
        self.plot_builder.save_figure(ax.figure, "hyperfine_speedup.png")
        return hf2

    def visualize_hyperfine_split(self, hf: pd.DataFrame, threshold_ms: float) -> None:
        """Create split visualizations for fast/slow benchmarks."""
        if hf.empty:
            print("No data for hyperfine split plot.")
            return

        fast = hf.query("mean_ms < @threshold_ms")
        slow = hf.query("mean_ms >= @threshold_ms")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

        # Plot fast benchmarks
        self.plot_builder.create_bar_plot(
            df=fast,
            x="benchmark",
            y="mean_ms",
            yerr="stddev_ms",
            title=f"Fast benchmarks (< {threshold_ms} ms)",
            ylabel="Mean runtime (ms)",
            ax=ax1,
        )

        # Plot slow benchmarks
        self.plot_builder.create_bar_plot(
            df=slow,
            x="benchmark",
            y="mean_ms",
            yerr="stddev_ms",
            title=f"Slow benchmarks (≥ {threshold_ms} ms) – Log Scale",
            ylabel="Mean runtime (ms)",
            log_y=True,
            ax=ax2,
            color="salmon",
        )

        self.plot_builder.save_figure(fig, f"hyperfine_split_{int(threshold_ms)}ms.png")

    def visualize_hyperfine_violin(self) -> pd.DataFrame:
        """Create violin plots from Hyperfine per-run data."""
        df = self.data_loader.load_hyperfine_json()
        if df.empty:
            return df

        ax = self.plot_builder.create_violin_plot(
            df=df,
            x="benchmark",
            y="runtime_ms",
            log_scale=True,
            title="Hyperfine: Runtime distributions (ms) – Log Scale",
            ylabel="Runtime (ms)",
        )
        self.plot_builder.save_figure(ax.figure, "hyperfine_runtimes_violin.png")
        return df

    def visualize_violin_symlog_split(
        self,
        df: pd.DataFrame,
        threshold_ms: float
    ) -> None:
        """Create enhanced violin plots with symlog scale and split view."""
        if df.empty:
            print("No data for enhanced violin plots.")
            return

        fig1, ax1 = plt.subplots(figsize=self.config.fig_size)
        self.plot_builder.create_violin_plot(
            df=df,
            x="benchmark",
            y="runtime_ms",
            symlog=True,
            linthresh=threshold_ms/10,
            title="Runtime distributions (ms) – SymLog Scale",
            ylabel="Runtime (ms)",
            ax=ax1,
        )
        self.plot_builder.save_figure(fig1, "hyperfine_runtimes_symlog.png")

        fast = df.query("runtime_ms < @threshold_ms")
        slow = df.query("runtime_ms >= @threshold_ms")

        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

        self.plot_builder.create_violin_plot(
            df=fast,
            x="benchmark",
            y="runtime_ms",
            title=f"Fast benchmarks (< {threshold_ms} ms)",
            ylabel="Runtime (ms)",
            ax=ax2,
        )

        self.plot_builder.create_violin_plot(
            df=slow,
            x="benchmark",
            y="runtime_ms",
            log_scale=True,
            title=f"Slow benchmarks (≥ {threshold_ms} ms) – Log Scale",
            ylabel="Runtime (ms)",
            ax=ax3,
        )

        self.plot_builder.save_figure(fig2, "hyperfine_runtimes_split.png")

    def visualize_gnutime(self) -> pd.DataFrame:
        """Create visualizations from GNU time data."""
        df = self.data_loader.load_gnutime()
        if df.empty:
            return df

        fig = plt.figure(figsize=self.config.fig_size)
        ax = sns.barplot(x="benchmark", y="max_rss_mb", data=df)
        ax.set_yscale("log")
        ax.set_title("Peak Memory Usage (MB) – Log Scale")
        ax.set_ylabel("Peak RSS (MB)")
        ax.tick_params(axis="x", rotation=45)
        self.plot_builder.save_figure(fig, "gnu_time_memory.png")
        return df

    def visualize_perf(self) -> pd.DataFrame:
        """Create visualizations from perf stat data."""
        pivot = self.data_loader.load_perf()
        if pivot.empty:
            return pivot

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for ax, col in zip(axes, pivot.columns):
            pivot[col].plot.bar(ax=ax, legend=False)
            ax.set_yscale("log")
            ax.set_title(col)
            ax.set_ylabel(f"{col} – Log Scale")
            ax.tick_params(axis="x", rotation=90)

        fig.suptitle("perf stat: Hardware Counters – Log Scale")
        self.plot_builder.save_figure(fig, "perf_stat_counters.png")
        return pivot

    def visualize_perf_heatmap(self, pivot: pd.DataFrame) -> pd.DataFrame:
        """Create heatmap visualization from perf stat data."""
        if pivot.empty:
            print("No data for perf heatmap.")
            return pivot

        normed = pivot.divide(pivot.max(), axis=1)
        fig = plt.figure(figsize=(12, max(4, len(normed) * 0.5)))
        ax = sns.heatmap(
            normed,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor="gray",
            cbar_kws={"label": "Normalized value (0–1)"},
        )
        ax.set_title("Normalized perf counters heatmap")
        ax.set_ylabel("Benchmark")
        ax.set_xlabel("Event")
        self.plot_builder.save_figure(fig, "perf_counters_heatmap.png")
        return normed

    def save_dataframes(
        self,
        hyperfine: pd.DataFrame,
        hyperfine_runs: pd.DataFrame,
        gnutime: pd.DataFrame,
        perf: pd.DataFrame
    ) -> None:
        """Save processed dataframes to CSV files."""
        if not hyperfine.empty:
            hyperfine.to_csv(self.config.output_dir / "hyperfine_summary.csv", index=False)
        if not hyperfine_runs.empty:
            hyperfine_runs.to_csv(self.config.output_dir / "hyperfine_runtimes_violin.csv", index=False)
        if not gnutime.empty:
            gnutime.to_csv(self.config.output_dir / "gnutime_summary.csv", index=False)
        if not perf.empty:
            perf.to_csv(self.config.output_dir / "perf_summary.csv")

    def run_visualization(self, threshold_ms: float) -> None:
        """Run the full visualization pipeline."""
        print("Parsing Hyperfine results...")
        hf = self.visualize_hyperfine()

        print("Generating Hyperfine split plot…")
        self.visualize_hyperfine_split(hf, threshold_ms)

        print("Generating Hyperfine speedup plot...")
        self.visualize_hyperfine_speedup(hf)

        print("Parsing Hyperfine per-run results...")
        hfj = self.visualize_hyperfine_violin()
        self.visualize_violin_symlog_split(hfj, threshold_ms=threshold_ms)

        print("Parsing GNU time results...")
        gt = self.visualize_gnutime()

        print("Parsing perf stat results...")
        perf = self.visualize_perf()

        print("Generating perf heatmap...")
        self.visualize_perf_heatmap(perf)

        self.save_dataframes(hf, hfj, gt, perf)

        print("Analysis complete. See outputs in:", self.config.output_dir)


def main() -> None:
    """Main entry point for the benchmark visualization tool."""
    parser = argparse.ArgumentParser(description="Process and plot benchmark results.")
    parser.add_argument(
        "--split-threshold",
        type=float,
        default=500.0,
        help="Threshold (ms) to split fast vs slow benchmarks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="benchmark_results",
        help="Directory for output files",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for saved figures",
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    visualizer = BenchmarkVisualizer(config)
    visualizer.run_visualization(args.split_threshold)


if __name__ == "__main__":
    main()
