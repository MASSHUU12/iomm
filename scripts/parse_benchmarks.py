#!/usr/bin/env python3

import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

OUT_DIR = "benchmark_results"

def parse_hyperfine() -> pd.DataFrame:
    """Parse all Hyperfine CSV files and plot runtime means with error bars (log scale)."""
    csv_files = glob.glob(os.path.join(OUT_DIR, "*_time.csv"))
    dfs = []
    for csvfile in csv_files:
        name = os.path.basename(csvfile).replace("_time.csv", "")
        df = pd.read_csv(csvfile)
        df["benchmark"] = name
        df["mean_ms"] = df["mean"] * 1000
        df["stddev_ms"] = df["stddev"] * 1000
        dfs.append(df)
    if not dfs:
        print("No Hyperfine results found.")
        return pd.DataFrame()

    all_hf = pd.concat(dfs, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        all_hf["benchmark"],
        all_hf["mean_ms"],
        yerr=all_hf["stddev_ms"],
        capsize=5,
        color="skyblue"
    )
    ax.set_yscale('log')
    ax.set_title("Hyperfine: Mean runtimes with stddev (ms) – Log Scale")
    ax.set_ylabel("Mean runtime (ms) – Log Scale")
    ax.set_xlabel("Benchmark")
    ax.tick_params(axis='x', rotation=45, labelrotation=45)

    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mtick.NullFormatter())
    ax.grid(which='major', axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "hyperfine_means.png")
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

    return all_hf

def parse_hyperfine_speedup(all_hf: pd.DataFrame) -> pd.DataFrame:
    """
    Plot relative speedups (baseline / mean_ms) so everything is on a common '×-speed' scale.
    """
    if all_hf.empty:
        print("No data for hyperfine speedup.")
        return all_hf

    baseline = all_hf["mean_ms"].min()
    all_hf = all_hf.copy()
    all_hf["speedup"] = baseline / all_hf["mean_ms"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        all_hf["benchmark"],
        all_hf["speedup"],
        color="skyblue"
    )
    ax.set_title("Hyperfine: Relative Speedup over Fastest (×)")
    ax.set_ylabel("Speedup factor")
    ax.set_xlabel("Benchmark")

    ax.tick_params(axis='x', rotation=45, labelrotation=45)
    ax.grid(which='major', axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "hyperfine_speedup.png")
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

    return all_hf

def parse_hyperfine_json() -> pd.DataFrame:
    json_files = glob.glob(os.path.join(OUT_DIR, "*_time.json"))
    all_runs = []
    for jsonfile in json_files:
        name = os.path.basename(jsonfile).replace("_time.json", "")
        with open(jsonfile) as f:
            data = json.load(f)
            for result in data.get("results", []):
                times = result.get("times")
                if times:
                    for t in times:
                        all_runs.append({"benchmark": name, "runtime_ms": t * 1000})
    if all_runs:
        df = pd.DataFrame(all_runs)
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        sns.violinplot(x="benchmark", y="runtime_ms", data=df, ax=ax)
        ax.set_yscale('log')
        plt.title("Hyperfine: Runtime distributions (ms) - Log Scale")
        plt.ylabel("Runtime (ms) - Log Scale")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "hyperfine_runtimes_violin.png"))
        print("Saved: hyperfine_runtimes_violin.png")
        return df
    else:
        print("No per-run timing data found in Hyperfine JSON")
        return pd.DataFrame()

def parse_gnutime() -> pd.DataFrame:
    """Parse GNU time CSV file and plot CPU/memory usage."""
    csvfile = os.path.join(OUT_DIR, "cpu_mem.csv")
    if not os.path.exists(csvfile):
        print("No GNU time results found.")
        return pd.DataFrame()
    df = pd.read_csv(csvfile, header=None, names=["benchmark", "elapsed_s", "user_s", "sys_s", "max_rss_kb"])
    df["max_rss_mb"] = df["max_rss_kb"] / 1024
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    sns.barplot(x="benchmark", y="max_rss_mb", data=df, ax=ax)
    ax.set_yscale('log')
    plt.title("Peak Memory Usage (MB) - Log Scale")
    plt.ylabel("Peak RSS (MB) - Log Scale")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "gnu_time_memory.png"))
    print("Saved: gnu_time_memory.png")
    return df

def parse_perf() -> pd.DataFrame:
    """Parse perf stat CSV files and plot hardware counter results."""
    import csv

    csv_files = glob.glob(os.path.join(OUT_DIR, "*_perf.csv"))
    dfs = []
    for csvfile in csv_files:
        name = os.path.basename(csvfile).replace("_perf.csv", "")
        perf_rows = []
        with open(csvfile, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                # Defensive: skip rows that are too short
                if len(row) < 3:
                    continue
                value = row[0]
                event = row[2]
                # Only keep rows where value and event look valid
                if value and event:
                    try:
                        value = float(value)
                    except ValueError:
                        continue
                    perf_rows.append({"benchmark": name, "event": event, "value": value})
        if perf_rows:
            dfs.append(pd.DataFrame(perf_rows))
    if dfs:
        all_perf = pd.concat(dfs, ignore_index=True)
        all_perf_pivot = all_perf.pivot(index="benchmark", columns="event", values="value")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for i, col in enumerate(all_perf_pivot.columns):
            if i < len(axes):
                ax = axes[i]
                all_perf_pivot[col].plot(kind="bar", ax=ax, legend=False)
                ax.set_title(col)
                ax.set_yscale('log')
                ax.set_ylabel(f"{col} - Log Scale")
                ax.tick_params(axis='x', rotation=90)

        plt.suptitle("perf stat: Hardware Counters - Log Scale")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(OUT_DIR, "perf_stat_counters.png"))
        print("Saved: perf_stat_counters.png")
        return all_perf_pivot
    else:
        print("No perf stat results found.")
        return pd.DataFrame()

def parse_perf_heatmap(all_perf_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Given the perf pivot table (benchmarks × events), normalize each event column to [0–1]
    and plot a heatmap of normalized counter values.
    """
    if all_perf_pivot.empty:
        print("No data for perf heatmap.")
        return all_perf_pivot

    normed = all_perf_pivot.divide(all_perf_pivot.max(axis=0), axis=1)
    plt.figure(figsize=(12, max(4, len(normed) * 0.5)))
    ax = sns.heatmap(
        normed,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Normalized value (0–1)"}
    )
    ax.set_title("Normalized perf counters heatmap")
    ax.set_ylabel("Benchmark")
    ax.set_xlabel("Event")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "perf_counters_heatmap.png")
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

    return normed


def main() -> None:
    print("Parsing Hyperfine results...")
    hf = parse_hyperfine()

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

    # Save summary tables
    if not hf.empty:
        hf.to_csv(os.path.join(OUT_DIR, "hyperfine_summary.csv"), index=False)
    if not hfj.empty:
        hfj.to_csv(os.path.join(OUT_DIR, "hyperfine_runtimes_violin.csv"), index=False)
    if not gt.empty:
        gt.to_csv(os.path.join(OUT_DIR, "gnutime_summary.csv"), index=False)
    if not perf.empty:
        perf.to_csv(os.path.join(OUT_DIR, "perf_summary.csv"))

    print("Analysis complete. See output images and CSVs in:", OUT_DIR)

if __name__ == "__main__":
    main()
