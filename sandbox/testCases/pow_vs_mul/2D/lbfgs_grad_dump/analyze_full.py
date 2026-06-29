"""Analyze iter-0 pow vs mul disagreement across the r1..r4 tag sweep."""

import glob
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAD_DIR = os.path.join(BASE_DIR, "gradients")

# Tag -> (NX = NY, nsteps, np). NZ = 2 fixed for all tags;
# nsteps = 1000 fixed for all tags (per attaway_r*.job nsteps_for_tag).
# NX, NY come from meshes/mesh_r*.yaml; np comes from attaway_r*.job np_for_tag.
TAGS = [
    ("r1",  45, 1000,  25),
    ("r2",  90, 1000,  36),
    ("r3", 135, 1000,  36),
    ("r4", 180, 1000, 144),
    ("r5", 250, 1000, 250),
]
NZ_VALUE = 2


def read_mm_dense(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while lines[i].startswith("%"):
        i += 1
    nrows = int(lines[i].split()[0])
    return np.array([float(x) for x in lines[i + 1 : i + 1 + nrows]], dtype=np.float64)


def list_gradient_paths(tag, form):
    prefix = f"grad_{tag}_{form}"
    pattern = os.path.join(GRAD_DIR, prefix + ".field.*.mm")
    return sorted(
        glob.glob(pattern), key=lambda p: int(p.split(".field.")[-1].split(".")[0])
    )


def load_gradient(tag, form):
    paths = list_gradient_paths(tag, form)
    if not paths:
        return None
    return np.concatenate([read_mm_dense(path) for path in paths])


_PAIR_CACHE = {}


def load_pair(tag):
    # Skip rows with mismatched counts (partial runs) instead of crashing on
    # a broadcast error; pow and mul must have stepped to the same time.
    if tag in _PAIR_CACHE:
        return _PAIR_CACHE[tag]
    mul_paths = list_gradient_paths(tag, "mul")
    pow_paths = list_gradient_paths(tag, "pow")
    if not mul_paths or not pow_paths:
        _PAIR_CACHE[tag] = (None, None)
        return None, None
    if len(mul_paths) != len(pow_paths):
        print(
            f"[warn] {tag}: mul has {len(mul_paths)} .field.*.mm, "
            f"pow has {len(pow_paths)} -- skipping (one or both runs are partial)"
        )
        _PAIR_CACHE[tag] = (None, None)
        return None, None
    grad_mul = np.concatenate([read_mm_dense(p) for p in mul_paths])
    grad_pow = np.concatenate([read_mm_dense(p) for p in pow_paths])
    _PAIR_CACHE[tag] = (grad_mul, grad_pow)
    return grad_mul, grad_pow


def disagreement_ratio(tag):
    grad_mul, grad_pow = load_pair(tag)
    if grad_mul is None:
        return None
    norm_mul = np.linalg.norm(grad_mul)
    norm_pow = np.linalg.norm(grad_pow)
    if norm_mul == 0.0:
        return None
    return (norm_pow / norm_mul) - 1.0


def relative_difference_norm(tag):
    grad_mul, grad_pow = load_pair(tag)
    if grad_mul is None:
        return None
    norm_mul = np.linalg.norm(grad_mul)
    if norm_mul == 0.0:
        return None
    return np.linalg.norm(grad_pow - grad_mul) / norm_mul


def print_markdown_table(title, formatter, value_fn):
    print(title)
    print()
    print("| tag | NX=NY | NZ | N_t |  np | value |")
    print("|----:|------:|---:|----:|----:|------:|")
    for tag, nxy, nt, np_val in TAGS:
        value = value_fn(tag)
        cell = "-" if value is None else formatter(value)
        print(f"| {tag} | {nxy} | {NZ_VALUE} | {nt} | {np_val} | {cell} |")
    print()


print("Tag sweep summary (rows skipped with '-' have no gradient data on disk)")
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
