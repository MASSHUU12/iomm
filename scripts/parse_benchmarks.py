#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

OUT_DIR = Path("benchmark_results")
OUT_DIR.mkdir(exist_ok=True)


def save_fig(fig: plt.Figure, name: str) -> None:
    """Tighten layout, save to OUT_DIR, and print status."""
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path)
    print(f"Saved: {path}")


def bar_plot(
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
    """
    Draw a bar plot (with optional yerr) on the given Axes, or create a new one.
    """
    fig_kwargs = fig_kwargs or {"figsize": (10, 6)}
    if ax is None:
        fig, ax = plt.subplots(**fig_kwargs)
    else:
        fig = ax.figure

    ax.bar(df[x], df[y], yerr=df[yerr] if yerr else None, capsize=5, color=color)

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


def read_hyperfine_csvs() -> pd.DataFrame:
    """Read all *_time.csv under OUT_DIR, annotate and return a single DataFrame."""
    files = sorted(OUT_DIR.glob("*_time.csv"))
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


def parse_hyperfine() -> pd.DataFrame:
    hf = read_hyperfine_csvs()
    if hf.empty:
        return hf

    ax = bar_plot(
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
    save_fig(ax.figure, "hyperfine_means.png")
    return hf


def parse_hyperfine_speedup(hf: pd.DataFrame) -> pd.DataFrame:
    if hf.empty:
        print("No data for hyperfine speedup.")
        return hf

    baseline = hf["mean_ms"].min()
    hf2 = hf.assign(speedup=lambda d: baseline / d["mean_ms"])

    ax = bar_plot(
        hf2,
        x="benchmark",
        y="speedup",
        title="Hyperfine: Relative Speedup over Fastest (×)",
        ylabel="Speedup factor",
    )
    save_fig(ax.figure, "hyperfine_speedup.png")
    return hf2


def parse_hyperfine_split(hf: pd.DataFrame, threshold_ms: float = 500.0) -> None:
    if hf.empty:
        print("No data for hyperfine split plot.")
        return

    fast = hf.query("mean_ms < @threshold_ms")
    slow = hf.query("mean_ms >= @threshold_ms")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    bar_plot(
        fast,
        x="benchmark",
        y="mean_ms",
        yerr="stddev_ms",
        title=f"Fast benchmarks (< {threshold_ms} ms)",
        ylabel="Mean runtime (ms)",
        ax=ax1,
    )
    bar_plot(
        slow,
        x="benchmark",
        y="mean_ms",
        yerr="stddev_ms",
        title=f"Slow benchmarks (≥ {threshold_ms} ms) – Log Scale",
        ylabel="Mean runtime (ms)",
        log_y=True,
        fmt_y_major=True,
        fmt_y_minor=True,
        color="salmon",
        ax=ax2,
    )

    save_fig(fig, f"hyperfine_split_{int(threshold_ms)}ms.png")


def parse_hyperfine_json() -> pd.DataFrame:
    files = sorted(OUT_DIR.glob("*_time.json"))
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

    df = pd.DataFrame(runs)
    fig = plt.figure(figsize=(10, 6))
    ax = sns.violinplot(x="benchmark", y="runtime_ms", data=df)
    ax.set_yscale("log")
    ax.set_title("Hyperfine: Runtime distributions (ms) – Log Scale")
    ax.set_ylabel("Runtime (ms)")
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, "hyperfine_runtimes_violin.png")
    return df


def parse_gnutime() -> pd.DataFrame:
    path = OUT_DIR / "cpu_mem.csv"
    if not path.exists():
        print("No GNU time results found.")
        return pd.DataFrame()

    df = pd.read_csv(path, header=None, names=["benchmark", "elapsed_s", "user_s", "sys_s", "max_rss_kb"])
    df["max_rss_mb"] = df["max_rss_kb"] / 1024

    fig = plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="benchmark", y="max_rss_mb", data=df)
    ax.set_yscale("log")
    ax.set_title("Peak Memory Usage (MB) – Log Scale")
    ax.set_ylabel("Peak RSS (MB)")
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, "gnu_time_memory.png")
    return df


def parse_perf() -> pd.DataFrame:
    files = sorted(OUT_DIR.glob("*_perf.csv"))
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
    pivot = perf.pivot(index="benchmark", columns="event", values="value")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, pivot.columns):
        pivot[col].plot.bar(ax=ax, legend=False)
        ax.set_yscale("log")
        ax.set_title(col)
        ax.set_ylabel(f"{col} – Log Scale")
        ax.tick_params(axis="x", rotation=90)

    fig.suptitle("perf stat: Hardware Counters – Log Scale")
    save_fig(fig, "perf_stat_counters.png")
    return pivot


def parse_perf_heatmap(pivot: pd.DataFrame) -> pd.DataFrame:
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
    save_fig(fig, "perf_counters_heatmap.png")
    return normed


def main(threshold_ms: float) -> None:
    print("Parsing Hyperfine results...")
    hf = parse_hyperfine()

    print("Generating Hyperfine split plot…")
    parse_hyperfine_split(hf, threshold_ms)

    print("Generating Hyperfine speedup plot...")
    parse_hyperfine_speedup(hf)

    print("Parsing Hyperfine per-run results...")
    hfj = parse_hyperfine_json()

    print("Parsing GNU time results...")
    gt = parse_gnutime()

    print("Parsing perf stat results...")
    perf = parse_perf()

    print("Generating perf heatmap...")
    parse_perf_heatmap(perf)

    if not hf.empty:
        hf.to_csv(OUT_DIR / "hyperfine_summary.csv", index=False)
    if not hfj.empty:
        hfj.to_csv(OUT_DIR / "hyperfine_runtimes_violin.csv", index=False)
    if not gt.empty:
        gt.to_csv(OUT_DIR / "gnutime_summary.csv", index=False)
    if not perf.empty:
        perf.to_csv(OUT_DIR / "perf_summary.csv")

    print("Analysis complete. See outputs in:", OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and plot benchmark results.")
    parser.add_argument(
        "--split-threshold",
        type=float,
        default=500.0,
        help="Threshold (ms) to split fast vs slow benchmarks",
    )
    args = parser.parse_args()
    main(args.split_threshold)
