#!/usr/bin/env python3
"""
Plot and summarize the regularization weight scan results.

Reuses the log parser from lbfgs_test/plot_qn_status.py.
Run from the reg_weight_scan/ directory:

    python3 plot_results.py
    python3 plot_results.py --root logs --show
    python3 plot_results.py --root logs --output my_plot.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

# Reuse the existing parser/plotter from lbfgs_test
LBFGS_DIR = Path(__file__).resolve().parent.parent / "lbfgs_test"
sys.path.insert(0, str(LBFGS_DIR))

from plot_qn_status import (
    collect_data,
    make_plot,
    print_objective_table,
)


def main() -> None:
    import argparse

    default_root = Path(__file__).resolve().parent / "logs"
    parser = argparse.ArgumentParser(
        description="Plot regularization weight scan results."
    )
    parser.add_argument(
        "--root", type=Path, default=default_root,
        help="Directory containing mrhyde_*.log files.",
    )
    parser.add_argument(
        "--pattern", default="mrhyde_*.log",
        help="Log filename pattern.",
    )
    parser.add_argument(
        "--output", type=Path, default=default_root / "qn_status_comparison.pdf",
        help="Output plot path.",
    )
    parser.add_argument("--show", action="store_true", help="Show plot window.")
    parser.add_argument("--no-table", action="store_true", help="Skip table output.")
    args = parser.parse_args()

    data, path_by_label = collect_data(args.root, args.pattern)

    if not args.no_table:
        print_objective_table(data, nsamples=8)

    # Write summary.md
    summary_path = args.root / "summary.md"
    import io
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_objective_table(data, nsamples=8)
    sys.stdout = old_stdout
    summary_path.write_text(buf.getvalue())
    print(f"Saved summary: {summary_path}")

    make_plot(data, path_by_label, args.output, args.show)
    print(f"Saved plot: {args.output}")
    print(f"Parsed datasets: {len(data)}")


if __name__ == "__main__":
    main()
