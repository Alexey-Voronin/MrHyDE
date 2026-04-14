#!/usr/bin/env python3
"""
Plot Quasi-Newton status metrics from mrhyde log files.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple


METRICS = ["value", "gnorm", "snorm", "alpha", "#fval"]
# Okabe-Ito palette
OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LOG_SCALE_METRICS = {"value", "gnorm", "snorm"}
HEADER_TOKENS = ["iter", "value", "gnorm", "snorm", "alpha", "#fval"]


def parse_number(token: str) -> float:
    if token == "---":
        return math.nan
    return float(token)


def parse_qn_rows(log_path: Path) -> Dict[str, List[float]]:
    series = {"iter": []}
    for metric in METRICS:
        series[metric] = []

    in_table = False
    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not in_table:
                if all(tok in line for tok in HEADER_TOKENS):
                    in_table = True
                continue

            if not line:
                continue
            if line.startswith("Optimization Terminated"):
                in_table = False
                continue

            parts = line.split()
            if len(parts) < 9:
                in_table = False
                continue
            if not parts[0].isdigit():
                in_table = False
                continue

            series["iter"].append(float(parts[0]))
            series["value"].append(parse_number(parts[1]))
            series["gnorm"].append(parse_number(parts[2]))
            series["snorm"].append(parse_number(parts[3]))
            series["alpha"].append(parse_number(parts[4]))
            series["#fval"].append(parse_number(parts[5]))

    return series


def collect_data(
    root_dir: Path, pattern: str
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Path]]:
    all_logs = sorted(root_dir.rglob(pattern))
    if not all_logs:
        raise FileNotFoundError(
            f"No log files matched pattern '{pattern}' under '{root_dir}'."
        )

    data: Dict[str, Dict[str, List[float]]] = {}
    path_by_label: Dict[str, Path] = {}
    for log_path in all_logs:
        series = parse_qn_rows(log_path)
        if series["iter"]:
            data[log_path.name] = series
            path_by_label[log_path.name] = log_path

    if not data:
        raise RuntimeError(
            "Log files were found, but no Quasi-Newton table rows were parsed."
        )
    return data, path_by_label


def make_plot(
    data: Dict[str, Dict[str, List[float]]],
    path_by_label: Dict[str, Path],
    output_path: Path,
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate plots. Install it with "
            "'python3 -m pip install matplotlib'."
        ) from exc

    fig, axes = plt.subplots(1, len(METRICS), figsize=(20, 4.8), sharex=True)
    if len(METRICS) == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    for series_idx, label in enumerate(sorted(data.keys())):
        series = data[label]
        x = series["iter"]
        log_path = path_by_label[label]
        no_scaling = "no_scaling" in str(log_path)
        if no_scaling:
            plot_kw = {"linewidth": 1.8, "color": "black", "label": label}
        else:
            markevery = max(1, len(x) // 25)
            plot_kw = {
                "linewidth": 1.8,
                "color": OKABE_ITO[series_idx % len(OKABE_ITO)],
                "marker": SERIES_MARKERS[series_idx % len(SERIES_MARKERS)],
                "markersize": 6,
                "markevery": markevery,
                "label": label,
            }
        for idx, metric in enumerate(METRICS):
            y = series[metric]
            (line,) = axes[idx].plot(x, y, **plot_kw)
            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(label)

    for idx, metric in enumerate(METRICS):
        axes[idx].set_title(metric)
        axes[idx].set_xlabel("iter")
        axes[idx].set_ylabel(metric)
        if metric in LOG_SCALE_METRICS:
            axes[idx].set_yscale("log")
        axes[idx].grid(True, which="major", color="lightgray", linestyle="-")
        axes[idx].minorticks_on()
        axes[idx].grid(True, which="minor", color="lightgray", linestyle=":")

    fig.suptitle("Quasi-Newton Status Comparison")
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0.0, 0.10, 1.0, 0.93])
    fig.savefig(output_path, dpi=250)

    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Parse Quasi-Newton status output and plot metrics from mrhyde logs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Root directory to search recursively for log files.",
    )
    parser.add_argument(
        "--pattern",
        default="mrhyde_*.log",
        help="Recursive log filename pattern to match.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "qn_status_comparison.pdf",
        help="Output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window in addition to saving the figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, path_by_label = collect_data(args.root, args.pattern)
    make_plot(data, path_by_label, args.output, args.show)
    print(f"Saved plot: {args.output}")
    print(f"Parsed datasets: {len(data)}")


if __name__ == "__main__":
    main()
