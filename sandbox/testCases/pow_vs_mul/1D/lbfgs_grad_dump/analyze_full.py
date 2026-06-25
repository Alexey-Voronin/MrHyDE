"""Analyze iter-0 pow vs mul disagreement on an NZ x N_t matrix at np=6."""

import glob
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAD_DIR = os.path.join(BASE_DIR, "gradients")

NZ_VALUES = [44, 88, 132, 176]
NT_VALUES = [400, 800, 1200, 1600]
NP_VALUE = 6


def read_mm_dense(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while lines[i].startswith("%"):
        i += 1
    nrows = int(lines[i].split()[0])
    return np.array([float(x) for x in lines[i + 1 : i + 1 + nrows]], dtype=np.float64)


def load_gradient(nz, nt, form):
    prefix = f"grad_nz{nz}_nt{nt}_np{NP_VALUE}_{form}"
    pattern = os.path.join(GRAD_DIR, prefix + ".field.*.mm")
    paths = sorted(
        glob.glob(pattern), key=lambda p: int(p.split(".field.")[-1].split(".")[0])
    )
    if not paths:
        return None
    return np.concatenate([read_mm_dense(path) for path in paths])


def load_pair(nz, nt):
    grad_mul = load_gradient(nz, nt, "mul")
    grad_pow = load_gradient(nz, nt, "pow")
    if grad_mul is None or grad_pow is None:
        return None, None
    return grad_mul, grad_pow


def disagreement_ratio(nz, nt):
    grad_mul, grad_pow = load_pair(nz, nt)
    if grad_mul is None:
        return None
    norm_mul = np.linalg.norm(grad_mul)
    norm_pow = np.linalg.norm(grad_pow)
    if norm_mul == 0.0:
        return None
    return (norm_pow / norm_mul) - 1.0


def relative_difference_norm(nz, nt):
    grad_mul, grad_pow = load_pair(nz, nt)
    if grad_mul is None:
        return None
    norm_mul = np.linalg.norm(grad_mul)
    if norm_mul == 0.0:
        return None
    return np.linalg.norm(grad_pow - grad_mul) / norm_mul


def format_row(values, formatter):
    formatted = []
    for value in values:
        if value is None:
            formatted.append("-")
        else:
            formatted.append(formatter(value))
    return formatted


def print_markdown_table(title, formatter, value_fn):
    print(title)
    print()
    print("| NZ \\\\ N_t | 400 | 800 | 1200 | 1600 |")
    print("|-----------:|----:|----:|-----:|-----:|")
    for nz in NZ_VALUES:
        row_values = [value_fn(nz, nt) for nt in NT_VALUES]
        row_cells = format_row(row_values, formatter)
        print(f"| {nz} | " + " | ".join(row_cells) + " |")
    print()


print(f"Matrix sweep summary at np={NP_VALUE}")
print()
print_markdown_table(
    "pow/mul - 1 = ||g_pow|| / ||g_mul|| - 1 (scientific notation)",
    lambda value: f"{value:+.2e}",
    disagreement_ratio,
)
print_markdown_table(
    "pow/mul - 1 as percent",
    lambda value: f"{value * 100.0:+.3f}%",
    disagreement_ratio,
)
print_markdown_table(
    "relative difference = ||g_pow - g_mul|| / ||g_mul||",
    lambda value: f"{value:.3e}",
    relative_difference_norm,
)
