#!/usr/bin/env python3
"""
Plot L-BFGS objective value from mrhyde log files.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List


OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]
SCALE_LINESTYLES = {"mul": "-", "pow": "--"}
HEADER_TOKENS = ["iter", "value", "gnorm", "snorm", "alpha", "#fval"]
R_TAG_META = {
    "r1": (44, 400),
    "r2": (88, 800),
    "r3": (132, 1200),
    "r4": (176, 1200),
}


def parse_number(token: str) -> float:
    if token == "---":
        return math.nan
    return float(token)


def parse_qn_value_rows(log_path: Path) -> Dict[str, List[float]]:
    series = {"iter": [], "value": []}
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
            if len(parts) < 2 or not parts[0].isdigit():
                in_table = False
                continue
            series["iter"].append(float(parts[0]))
            series["value"].append(parse_number(parts[1]))
    return series


def parse_r_tag(label: str) -> str:
    match = re.search(r"(?:^|[_\-])(r\d+)(?:[_\-\.]|$)", label.lower())
    if match:
        return match.group(1)
    return "r-unknown"


def parse_r_sort_key(rtag: str) -> tuple[int, int | str]:
    match = re.search(r"r(\d+)", rtag)
    if match:
        return (0, int(match.group(1)))
    return (1, rtag)


def parse_scale_tag(label: str) -> str:
    text = label.lower()
    if re.search(r"(?:^|[_\-])mul(?:[_\-\.]|$)", text):
        return "mul"
    if re.search(r"(?:^|[_\-])pow(?:[_\-\.]|$)", text):
        return "pow"
    return "unknown"


def metadata_for_label(label: str) -> tuple[int | None, int | None]:
    rtag = parse_r_tag(label)
    return R_TAG_META.get(rtag, (None, None))


def legend_entry(label: str) -> str:
    nz, nt = metadata_for_label(label)
    scale = parse_scale_tag(label)
    nz_text = f"{nz}" if nz is not None else "Nz?"
    nt_text = f"{nt}" if nt is not None else "Nt?"
    return f"({nz_text}, {nt_text}, {scale})"


def collect_data(root_dir: Path, pattern: str) -> Dict[str, Dict[str, List[float]]]:
    all_logs = sorted(root_dir.rglob(pattern))
    if not all_logs:
        raise FileNotFoundError(
            f"No log files matched pattern '{pattern}' under '{root_dir}'."
        )
    data: Dict[str, Dict[str, List[float]]] = {}
    for log_path in all_logs:
        series = parse_qn_value_rows(log_path)
        if series["iter"]:
            data[log_path.name] = series
    if not data:
        raise RuntimeError("Logs found, but no optimization rows parsed.")
    return data


def make_plot(data: Dict[str, Dict[str, List[float]]], output_path: Path, show: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required. Install with 'python3 -m pip install matplotlib'."
        ) from exc

    fig, ax = plt.subplots(1, 1, figsize=(12.0, 8.2), sharex=True)
    title_fs = 18
    label_fs = 15
    tick_fs = 13
    legend_fs = 13

    color_by_case: Dict[tuple[int | None, int | None], str] = {}
    cases = sorted(
        {metadata_for_label(label) for label in data.keys()},
        key=lambda case: (
            case[0] is None,
            case[0] if case[0] is not None else 10**9,
            case[1] is None,
            case[1] if case[1] is not None else 10**9,
        ),
    )
    for idx, case in enumerate(cases):
        color_by_case[case] = OKABE_ITO[idx % len(OKABE_ITO)]

    handle_by_label = {}
    for series_idx, label in enumerate(sorted(data.keys())):
        series = data[label]
        nz, nt = metadata_for_label(label)
        scale = parse_scale_tag(label)
        case = (nz, nt)
        (line,) = ax.plot(
            series["iter"],
            series["value"],
            linewidth=1.8,
            alpha=0.75,
            color=color_by_case.get(case, OKABE_ITO[series_idx % len(OKABE_ITO)]),
            linestyle=SCALE_LINESTYLES.get(scale, "-."),
            label=legend_entry(label),
        )
        handle_by_label[label] = line

    ax.set_title("L-BFGS objective value vs iter", fontsize=title_fs)
    ax.set_xlabel("iter", fontsize=label_fs)
    ax.set_ylabel("value", fontsize=label_fs)
    ax.tick_params(axis="both", which="both", labelsize=tick_fs)
    ax.set_yscale("log")
    ax.grid(True, which="major", color="lightgray", linestyle="-")
    ax.minorticks_on()
    ax.grid(True, which="minor", color="lightgray", linestyle=":")

    scale_order = {"mul": 0, "pow": 1, "unknown": 2}
    legend_labels = sorted(
        handle_by_label.keys(),
        key=lambda lbl: (
            metadata_for_label(lbl)[0] is None,
            metadata_for_label(lbl)[0] if metadata_for_label(lbl)[0] is not None else 10**9,
            metadata_for_label(lbl)[1] is None,
            metadata_for_label(lbl)[1] if metadata_for_label(lbl)[1] is not None else 10**9,
            scale_order.get(parse_scale_tag(lbl), 2),
            lbl,
        ),
    )
    legend_handles = [handle_by_label[lbl] for lbl in legend_labels]
    legend_text = [legend_entry(lbl) for lbl in legend_labels]
    ax.legend(
        legend_handles,
        legend_text,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        fontsize=legend_fs,
        ncol=4,
        title="(Nz, Nt, formulation)",
        title_fontsize=legend_fs,
    )
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 1.0])
    fig.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0.02)

    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot L-BFGS objective value from logs.")
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
    data = collect_data(args.root, args.pattern)
    make_plot(data, args.output, args.show)
    print(f"Saved plot: {args.output}")
    print(f"Parsed datasets: {len(data)}")


if __name__ == "__main__":
    main()
