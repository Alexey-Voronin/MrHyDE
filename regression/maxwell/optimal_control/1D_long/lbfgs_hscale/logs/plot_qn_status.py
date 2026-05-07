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


def final_objective_value(series: Dict[str, List[float]]) -> float:
    values = series.get("value", [])
    for v in reversed(values):
        if not math.isnan(v):
            return v
    return float("nan")


def objective_samples(series: Dict[str, List[float]], nsamples: int = 8) -> List[float]:
    values = series.get("value", [])
    m = len(values)
    if m < 2:
        return [math.nan] * nsamples
    last_idx = m - 1
    idxs = [int(math.floor(k * last_idx / nsamples)) for k in range(1, nsamples + 1)]
    samples = []
    for idx in idxs:
        if 0 <= idx < m:
            samples.append(values[idx])
        else:
            samples.append(math.nan)
    return samples


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


def _format_cell(x: float, width: int = 12) -> str:
    if math.isnan(x):
        return "---".rjust(width)
    return f"{x:{width}.1e}"


def print_objective_table(data: Dict[str, Dict[str, List[float]]], nsamples: int = 8) -> None:
    rows = []
    for label, series in data.items():
        final_val = final_objective_value(series)
        samples = objective_samples(series, nsamples=nsamples)
        rows.append((label, final_val, samples))

    def sort_key(row: tuple[str, float, List[float]]) -> tuple[int, float, str]:
        label, final_val, _samples = row
        nan_rank = 1 if math.isnan(final_val) else 0
        val = final_val if not math.isnan(final_val) else float("inf")
        return (nan_rank, val, label)

    rows.sort(key=sort_key)

    file_width = max(8, max(len(r[0]) for r in rows))
    header = ["file".ljust(file_width)] + [f"s{k}".rjust(12) for k in range(1, nsamples + 1)]
    print(" ".join(header))
    for label, _final_val, samples in rows:
        line = [label.ljust(file_width)] + [_format_cell(x, width=12) for x in samples]
        print(" ".join(line))


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

    handle_by_label = {}
    final_value_by_label = {}

    for series_idx, label in enumerate(sorted(data.keys())):
        series = data[label]
        final_value_by_label[label] = final_objective_value(series)
        x = series["iter"]
        log_path = path_by_label[label]
        no_scaling = "no_scaling" in str(log_path) or "noscal" in str(log_path)
        if no_scaling:
            plot_kw = {"linewidth": 1.8, "color": "black", "alpha": 0.7, "label": label}
        else:
            markevery = max(1, len(x) // 25)
            plot_kw = {
                "linewidth": 1.8,
                "alpha": 0.45,
                "color": OKABE_ITO[series_idx % len(OKABE_ITO)],
                "marker": SERIES_MARKERS[series_idx % len(SERIES_MARKERS)],
                "markersize": 4,
                "markeredgewidth": 0.6,
                "markevery": markevery,
                "label": label,
            }
        for idx, metric in enumerate(METRICS):
            y = series[metric]
            (line,) = axes[idx].plot(x, y, **plot_kw)
            if idx == 0:
                handle_by_label[label] = line

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
    legend_labels = sorted(
        handle_by_label.keys(),
        key=lambda lbl: (
            1 if math.isnan(final_value_by_label.get(lbl, float("nan"))) else 0,
            -(final_value_by_label.get(lbl, float("-inf"))),
            lbl,
        ),
    )
    legend_handles = [handle_by_label[lbl] for lbl in legend_labels]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        title="Sorted by final objective value (desc)",
    )
    fig.tight_layout(rect=[0.0, 0.15, 1.0, 0.93])
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
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Do not print the objective sampling table to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, path_by_label = collect_data(args.root, args.pattern)
    if not args.no_table:
        print_objective_table(data, nsamples=8)
    make_plot(data, path_by_label, args.output, args.show)
    print(f"Saved plot: {args.output}")
    print(f"Parsed datasets: {len(data)}")


if __name__ == "__main__":
    main()
