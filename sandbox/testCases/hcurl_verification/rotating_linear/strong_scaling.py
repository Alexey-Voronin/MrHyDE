"""
Strong scaling analysis for MrHyDE log files.

Analyzes how the most time-consuming routines scale with increasing
number of MPI tasks (strong scaling study).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parse_mrhyde_log import parse_log


def get_num_procs(workload_df):
    """Infer number of processors from workload DataFrame."""
    if workload_df.empty:
        return 1
    return workload_df["processor"].nunique()


def get_time_column(timing_df):
    """Determine which time column to use based on available columns."""
    if "GlobalTime" in timing_df.columns:
        return "GlobalTime"
    elif "MeanOverProcs" in timing_df.columns:
        return "MeanOverProcs"
    else:
        raise ValueError("No recognized time column in timing data")


def load_all_logs(log_files):
    """
    Load timing data from all log files.

    Returns:
        dict: {num_procs: timing_df} mapping processor count to timing data
    """
    results = {}
    for filepath in log_files:
        workload_df, timing_df = parse_log(filepath)
        num_procs = get_num_procs(workload_df)

        if timing_df.empty:
            print(f"Warning: No timing data in {filepath}, skipping")
            continue

        time_col = get_time_column(timing_df)
        timing_df = timing_df[["timer_name", time_col]].copy()
        timing_df.columns = ["timer_name", "time"]

        timing_df = timing_df[timing_df["time"] > 0]

        if num_procs in results:
            print(f"Warning: Duplicate processor count {num_procs}, "
                  f"overwriting with {filepath}")
        results[num_procs] = timing_df

    return results


def filter_top_routines(all_data, top_percent):
    """
    Filter to top N% routines by time from baseline (smallest proc count).

    Args:
        all_data: dict mapping num_procs to timing DataFrame
        top_percent: percentage of routines to keep (e.g., 10 for top 10%)

    Returns:
        list: timer names to include
    """
    baseline_procs = min(all_data.keys())
    baseline_df = all_data[baseline_procs]

    sorted_df = baseline_df.sort_values("time", ascending=False)
    n_keep = max(1, int(len(sorted_df) * top_percent / 100))
    top_timers = sorted_df.head(n_keep)["timer_name"].tolist()

    return top_timers


def build_timing_table(all_data, timer_names):
    """
    Build a table with routines as rows and processor counts as columns.

    Returns:
        pd.DataFrame with timer_name index and proc counts as columns
    """
    proc_counts = sorted(all_data.keys())
    rows = []

    for timer in timer_names:
        row = {"timer_name": timer}
        for nprocs in proc_counts:
            df = all_data[nprocs]
            match = df[df["timer_name"] == timer]
            if not match.empty:
                row[nprocs] = match["time"].values[0]
            else:
                row[nprocs] = np.nan
        rows.append(row)

    table = pd.DataFrame(rows).set_index("timer_name")
    return table


def compute_speedup(timing_table):
    """Compute speedup relative to smallest processor count."""
    baseline_col = timing_table.columns[0]
    speedup = timing_table.copy()
    for col in timing_table.columns:
        speedup[col] = timing_table[baseline_col] / timing_table[col]
    return speedup


def compute_efficiency(timing_table):
    """Compute parallel efficiency: T1 / (N * TN)."""
    baseline_col = timing_table.columns[0]
    baseline_procs = baseline_col
    efficiency = timing_table.copy()
    for col in timing_table.columns:
        nprocs = col
        scale = nprocs / baseline_procs
        efficiency[col] = (timing_table[baseline_col] /
                           (scale * timing_table[col])) * 100
    return efficiency


def print_tables(timing_table, speedup_table, efficiency_table):
    """Print tables to console."""
    print("\n" + "=" * 70)
    print("RAW TIMING (seconds)")
    print("=" * 70)
    print(timing_table.to_string(float_format=lambda x: f"{x:.4g}"))

    print("\n" + "=" * 70)
    print("SPEEDUP (relative to {} procs)".format(timing_table.columns[0]))
    print("=" * 70)
    print(speedup_table.to_string(float_format=lambda x: f"{x:.2f}"))

    print("\n" + "=" * 70)
    print("PARALLEL EFFICIENCY (%)")
    print("=" * 70)
    print(efficiency_table.to_string(float_format=lambda x: f"{x:.1f}"))


def shorten_timer_name(name, max_len=40):
    """Shorten timer name for plot labels."""
    name = name.replace("MrHyDE::", "")
    if len(name) > max_len:
        return name[:max_len-3] + "..."
    return name


def create_plots(timing_table, speedup_table, efficiency_table, output_path):
    """Create PDF with scaling plots."""
    proc_counts = list(timing_table.columns)
    timers = timing_table.index.tolist()
    short_names = [shorten_timer_name(t) for t in timers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ax1: Speedup vs Processors
    baseline = proc_counts[0]
    ideal_x = np.array(proc_counts)
    ideal_speedup = ideal_x / baseline
    ax1.plot(ideal_x, ideal_speedup, "k--", linewidth=2, label="Ideal")

    for timer, short in zip(timers, short_names):
        y = speedup_table.loc[timer].values
        ax1.plot(proc_counts, y, "o-", label=short, markersize=6)

    ax1.set_xlabel("Number of Processors", fontsize=12)
    ax1.set_ylabel("Speedup", fontsize=12)
    ax1.set_title("Strong Scaling: Speedup", fontsize=14)
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(proc_counts)

    # ax2: Parallel Efficiency
    x = np.arange(len(proc_counts))
    width = 0.8 / len(timers) if len(timers) > 0 else 0.8

    for i, (timer, short) in enumerate(zip(timers, short_names)):
        y = efficiency_table.loc[timer].values
        # Replace NaN with 0 for bar plotting
        y_plot = np.where(np.isnan(y), 0, y)
        offset = (i - len(timers) / 2 + 0.5) * width
        ax2.bar(x + offset, y_plot, width, label=short)

    ax2.axhline(y=100, color="k", linestyle="--", linewidth=1,
               label="Ideal (100%)")
    ax2.set_xlabel("Number of Processors", fontsize=12)
    ax2.set_ylabel("Parallel Efficiency (%)", fontsize=12)
    ax2.set_title("Strong Scaling: Parallel Efficiency", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(proc_counts)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(
        description="Strong scaling analysis for MrHyDE log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python strong_scaling.py np1.log np2.log np4.log --top-percent 10
        """
    )
    parser.add_argument(
        "logfiles",
        nargs="+",
        help="Log files to analyze (first file used as baseline for filtering)"
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=10,
        help="Percentage of top routines to show (default: 10)"
    )
    parser.add_argument(
        "--output",
        default="strong_scaling.pdf",
        help="Output PDF file path (default: strong_scaling.pdf)"
    )

    args = parser.parse_args()

    for f in args.logfiles:
        if not Path(f).exists():
            print(f"Error: File not found: {f}")
            sys.exit(1)

    print(f"Loading {len(args.logfiles)} log files...")
    all_data = load_all_logs(args.logfiles)

    if len(all_data) < 2:
        print("Error: Need at least 2 valid log files for scaling analysis")
        sys.exit(1)

    print(f"Found processor counts: {sorted(all_data.keys())}")

    top_timers = filter_top_routines(all_data, args.top_percent)
    print(f"Analyzing top {args.top_percent}% routines ({len(top_timers)} timers)")

    timing_table = build_timing_table(all_data, top_timers)
    speedup_table = compute_speedup(timing_table)
    efficiency_table = compute_efficiency(timing_table)

    print_tables(timing_table, speedup_table, efficiency_table)
    create_plots(timing_table, speedup_table, efficiency_table, args.output)


if __name__ == "__main__":
    main()

