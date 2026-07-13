#!/usr/bin/env python3
"""
Plot per-iteration Riesz energy diagnostics emitted by MrHyDE when
'riesz diagnostics: true' is set in the Analysis block.

Reads one or more CSV files of the form

  iter,gMg,gKg,zMz,zKz,ratio_g,ratio_z,alpha1,alpha2

and produces a single PDF with rows = refinement levels and columns = plot type:
  Col 1: gMg and gKg vs iter (log y)
  Col 2: zMz and zKz vs iter (log y)
  Col 3: ratio_g and ratio_z (log y), where
         ratio_g = (gKg)/(gMg), ratio_z = (zKz)/(zMz)

Within each subplot, all runs for that refinement level are overlaid.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_csv(path: Path) -> Tuple[List[str], List[List[float]]]:
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        rows: List[List[float]] = []
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue
            row: List[float] = []
            ok = True
            for p in parts:
                if p == "nan" or p == "":
                    row.append(math.nan)
                else:
                    try:
                        row.append(float(p))
                    except ValueError:
                        ok = False
                        break
            if ok:
                rows.append(row)
    return header, rows


def column(rows: List[List[float]], idx: int) -> List[float]:
    return [r[idx] for r in rows]


def parse_tag_and_run(stem: str) -> Tuple[str, str]:
    parts = stem.split("_")
    for idx, part in enumerate(parts):
        if re.fullmatch(r"r\d+", part):
            tag = part
            run_parts = parts[idx + 1 :]
            run = "_".join(run_parts) if run_parts else "default"
            return tag, run
    return "unknown", stem


def make_plot(
    data: Dict[str, Tuple[List[str], List[List[float]]]],
    output_path: Path,
    show: bool,
    drop_iterate_energy: bool,
    drop_iterate_ratio: bool,
) -> None:
    import matplotlib.pyplot as plt

    grouped: Dict[str, Dict[str, Tuple[List[str], List[List[float]]]]] = {}
    for label, payload in data.items():
        tag, run = parse_tag_and_run(label)
        grouped.setdefault(tag, {})[run] = payload

    def tag_sort_key(tag: str) -> Tuple[int, object]:
        m = re.fullmatch(r"r(\d+)", tag)
        if m:
            return (0, int(m.group(1)))
        return (1, tag)

    tags = sorted(grouped.keys(), key=tag_sort_key)

    if not tags:
        raise ValueError("No usable data sets to plot.")

    runs_all: List[str] = sorted({run for t in tags for run in grouped.get(t, {}).keys()})

    transposed_layout = drop_iterate_energy
    if transposed_layout:
        nrows = 2
        ncols = len(tags)
    else:
        nrows = len(tags)
        ncols = 3
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.5 * ncols, 3.8 * nrows),
        sharex=True,
    )
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [list(axes)]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    colorblind_palette = [
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#009E73",  # bluish green
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
    ]
    run_to_color: Dict[str, str] = {}
    color_idx = 0
    for run in runs_all:
        if "noscale" in run.lower():
            run_to_color[run] = "black"
        else:
            run_to_color[run] = colorblind_palette[color_idx % len(colorblind_palette)]
            color_idx += 1

    alpha_primary = 0.95
    alpha_secondary = 0.55
    linestyle_primary = "-"
    linestyle_secondary = "--"

    if transposed_layout:
        for col_idx, tag in enumerate(tags):
            runs = sorted(grouped.get(tag, {}).keys())
            if not runs:
                axes[0][col_idx].set_title(f"{tag}: no data")
                continue

            for run in runs:
                header, rows = grouped[tag][run]
                if not rows:
                    continue

                idx = {name: header.index(name) for name in header}
                it = column(rows, idx["iter"])
                gMg = column(rows, idx["gMg"])
                gKg = column(rows, idx["gKg"])
                ratio_g = column(rows, idx["ratio_g"])
                ratio_z = (
                    column(rows, idx["ratio_z"])
                    if "ratio_z" in idx
                    else [math.nan] * len(it)
                )

                color = run_to_color[run]

                ax = axes[0][col_idx]
                ax.plot(
                    it,
                    gMg,
                    color=color,
                    linestyle=linestyle_primary,
                    linewidth=1.2,
                    alpha=alpha_primary,
                    label=f"{run}: gMg",
                )
                ax.plot(
                    it,
                    gKg,
                    color=color,
                    linestyle=linestyle_secondary,
                    linewidth=1.2,
                    alpha=alpha_secondary,
                    label=f"{run}: gKg",
                )

                ax = axes[1][col_idx]
                ax.plot(
                    it,
                    ratio_g,
                    color=color,
                    linestyle=linestyle_primary,
                    linewidth=1.2,
                    alpha=alpha_primary,
                    label=f"{run}: ratio_g",
                )
                if not drop_iterate_ratio:
                    ax.plot(
                        it,
                        ratio_z,
                        color=color,
                        linestyle=linestyle_secondary,
                        linewidth=1.2,
                        alpha=alpha_secondary,
                        label=f"{run}: ratio_z",
                    )

        for r in range(nrows):
            for c in range(ncols):
                ax = axes[r][c]
                ax.set_yscale("log")
                ax.grid(True, which="major", color="lightgray")
                ax.minorticks_on()
                ax.grid(True, which="minor", color="lightgray", linestyle=":")
                ax.legend(loc="best", frameon=False, fontsize=8, ncol=2)

        for col_idx, tag in enumerate(tags):
            axes[0][col_idx].set_title(tag)
        axes[0][0].set_ylabel("gradient\nenergy")
        axes[1][0].set_ylabel("ratios")
    else:
        for row_idx, tag in enumerate(tags):
            runs = sorted(grouped.get(tag, {}).keys())
            if not runs:
                for c in range(ncols):
                    axes[row_idx][c].set_title(f"{tag}: no data")
                continue

            for run in runs:
                header, rows = grouped[tag][run]
                if not rows:
                    continue

                idx = {name: header.index(name) for name in header}
                it = column(rows, idx["iter"])
                gMg = column(rows, idx["gMg"])
                gKg = column(rows, idx["gKg"])
                zMz = column(rows, idx["zMz"])
                zKz = column(rows, idx["zKz"])
                ratio_g = column(rows, idx["ratio_g"])
                ratio_z = (
                    column(rows, idx["ratio_z"])
                    if "ratio_z" in idx
                    else [math.nan] * len(it)
                )

                color = run_to_color[run]

                ax = axes[row_idx][0]
                ax.plot(
                    it,
                    gMg,
                    color=color,
                    linestyle=linestyle_primary,
                    linewidth=1.2,
                    alpha=alpha_primary,
                    label=f"{run}: gMg",
                )
                ax.plot(
                    it,
                    gKg,
                    color=color,
                    linestyle=linestyle_secondary,
                    linewidth=1.2,
                    alpha=alpha_secondary,
                    label=f"{run}: gKg",
                )

                ax = axes[row_idx][1]
                ax.plot(
                    it,
                    zMz,
                    color=color,
                    linestyle=linestyle_primary,
                    linewidth=1.2,
                    alpha=alpha_primary,
                    label=f"{run}: zMz",
                )
                ax.plot(
                    it,
                    zKz,
                    color=color,
                    linestyle=linestyle_secondary,
                    linewidth=1.2,
                    alpha=alpha_secondary,
                    label=f"{run}: zKz",
                )

                ax = axes[row_idx][2]
                ax.plot(
                    it,
                    ratio_g,
                    color=color,
                    linestyle=linestyle_primary,
                    linewidth=1.2,
                    alpha=alpha_primary,
                    label=f"{run}: ratio_g",
                )
                if not drop_iterate_ratio:
                    ax.plot(
                        it,
                        ratio_z,
                        color=color,
                        linestyle=linestyle_secondary,
                        linewidth=1.2,
                        alpha=alpha_secondary,
                        label=f"{run}: ratio_z",
                    )

            for c in range(ncols):
                ax = axes[row_idx][c]
                ax.set_yscale("log")
                ax.grid(True, which="major", color="lightgray")
                ax.minorticks_on()
                ax.grid(True, which="minor", color="lightgray", linestyle=":")
                ax.legend(loc="best", frameon=False, fontsize=8, ncol=2)

            axes[row_idx][0].set_ylabel(f"{tag}\nenergy")

        axes[0][0].set_title(r"gradient energy: $g^T M g$ and $g^T K g$")
        axes[0][1].set_title(r"iterate energy: $z^T M z$ and $z^T K z$")
        if drop_iterate_ratio:
            axes[0][2].set_title(r"energy ratio: $\mathrm{ratio\_g}=(g^T K g)/(g^T M g)$")
        else:
            axes[0][2].set_title(
                r"energy ratios: $\mathrm{ratio\_g}=(g^T K g)/(g^T M g)$, "
                r"$\mathrm{ratio\_z}=(z^T K z)/(z^T M z)$"
            )

    for c in range(ncols):
        axes[nrows - 1][c].set_xlabel("iter")

    if transposed_layout:
        ratio_desc = "ratio_g only" if drop_iterate_ratio else "ratio_g and ratio_z"
        fig.suptitle(
            "Riesz diagnostics: rows=metric (gradient, ratio), "
            f"cols=refinement, colors=runs, ratios={ratio_desc}"
        )
    else:
        ratio_desc = "ratio_g only" if drop_iterate_ratio else "ratio_g and ratio_z"
        fig.suptitle(
            "Riesz energy diagnostics: rows=refinement, cols=metric, "
            f"colors=runs, ratios={ratio_desc}"
        )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot Riesz energy CSV files emitted by MrHyDE."
    )
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--pattern", default="mrhyde_riesz_diag*.csv")
    parser.add_argument("--output", type=Path,
                        default=default_root / "qn_riesz_diag.pdf")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--drop-iterate-energy",
        action="store_true",
        help="Drop iterate energy plots (zMz, zKz) and switch to rows=metric, cols=refinement.",
    )
    parser.add_argument(
        "--drop-iterate-ratio",
        action="store_true",
        help="Drop iterate ratio (ratio_z) from ratio subplots.",
    )
    parser.add_argument(
        "--drop-iterate",
        action="store_true",
        help="Shortcut for --drop-iterate-energy --drop-iterate-ratio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drop_iterate_energy = args.drop_iterate_energy or args.drop_iterate
    drop_iterate_ratio = args.drop_iterate_ratio or args.drop_iterate
    paths = sorted(args.root.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files matched '{args.pattern}' under '{args.root}'."
        )
    data: Dict[str, Tuple[List[str], List[List[float]]]] = {}
    for p in paths:
        label = p.stem
        data[label] = parse_csv(p)
        print(f"  {p.name}: {len(data[label][1])} rows")
    make_plot(
        data,
        args.output,
        args.show,
        drop_iterate_energy=drop_iterate_energy,
        drop_iterate_ratio=drop_iterate_ratio,
    )
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
