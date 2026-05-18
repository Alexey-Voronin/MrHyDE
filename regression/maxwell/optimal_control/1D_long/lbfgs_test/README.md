# LBFGS Metric-Norm Study Summary

## Workflow and Data Collection

1. Deck setup:
   - Study decks are stored in `rol_decks/`.
   - the rest of the decks are shared `input.yaml` and `other_decks`
2. Run collection:
   - `bash run_roldecks.sh` copies each deck to `input_rol.yaml` and launches `mrhyde`.
3. Diagnostics (top of each log):
   - `[MetricOpStats]` reports operator scale for `M`, `K`, and `H = alpha1*M + alpha2*K`.
     - Use `abs_row_sum` and `K_over_M` for magnitude comparison.
   - `MrHyDE vector contract check` calls `ROL::Vector::checkVector(...)` on both primal and dual vectors to validate vector-space/duality consistency, with tolerance `1e-8`.
   - Representative standard metric run (`direct`, `alpha1=alpha2=1e5`):
     - `M abs_row_sum_mean=6.59429e-4`
     - `K abs_row_sum_mean=1.93296e8`
     - `K_over_M abs_row_sum_ratio(mean)=2.93127e11` (`max=1.33333e11`)
     - `H abs_row_sum_mean=1.93296e8`
     - `max_dual_err` is typically `1e-15` to `1e-12` (well below `1e-8`)
4. Plot generation:
   - `plot_qn_status.py` generates `qn_status_comparison.pdf` in each study directory.

## Main Takeaways

- `K` is much larger than `M` in magnitude metrics (`K_over_M` about `1e11`), so stiffness dominance is real.

## Alpha Ratio Scan (`alpha_ratio`)

| File name | Metric norm | `alpha1` | `alpha2` | Final objective | Final gnorm | gnorm0/gnormN | Iteration |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_scaling | no_scaling | N/A | N/A | 4.32e-18 | 4.16e-04 | 5.85e+03 | 120 |
| s1e3_r1em2 | metric | 1.0e3 | 1.0e1 | 3.47e-17 | 2.26e-07 | 2.69e+03 | 51 |
| s1e3_r1e0 | metric | 1.0e3 | 1.0e3 | 8.76e-18 | 1.65e-08 | 3.69e+03 | 109 |
| s1e3_r1e2 | metric | 1.0e3 | 1.0e5 | 5.13e-18 | 3.53e-10 | 1.72e+04 | 120 |
| s1e5_r1em2 | metric | 1.0e5 | 1.0e3 | 6.24e-18 | 5.86e-09 | 1.04e+04 | 120 |
| s1e5_r1e0 | metric | 1.0e5 | 1.0e5 | 5.28e-18 | 4.80e-10 | 1.26e+04 | 120 |
| s1e5_r1e2 | metric | 1.0e5 | 1.0e7 | 1.20e-17 | 1.92e-10 | 3.17e+03 | 94 |
| s1e7_r1em2 | metric | 1.0e7 | 1.0e5 | 2.32e-17 | 4.39e-09 | 1.38e+03 | 68 |
| s1e7_r1e0 | metric | 1.0e7 | 1.0e7 | 7.29e-18 | 1.12e-10 | 5.43e+03 | 120 |
| s1e7_r1e2 | metric | 1.0e7 | 1.0e9 | 7.85e-18 | 1.28e-11 | 4.76e+03 | 120 |

PDF plot: [alpha_ratio/qn_status_comparison.pdf](alpha_ratio/qn_status_comparison.pdf)


`alpha2=0` performs onlt slightly worse than `no_scaling` but tracks it overall.  

## H0 Scaling Sweep (`H0_scaling/no_riesz`)

`HessVec Precond Mode: identity` with the L-BFGS initial-Hessian scale
`InitialHessian: alpha` varied. No Riesz map (`no_scaling`).

| File name | H0 (alpha) | Final objective | Final gnorm | gnorm0/gnormN | Iteration |
| --- | ---: | ---: | ---: | ---: | ---: |
| h0_auto | auto    | 4.32e-18 | 4.15e-04 | 5.85e+03 | 120 |
| h0_1e0  | 1.0e0   | 5.79e-15 | 9.16e-02 | 2.65e+01 | 50  |
| h0_1e4  | 1.0e4   | 4.42e-15 | 8.34e-02 | 2.92e+01 | 52  |
| h0_1e8  | 1.0e8   | 8.57e-16 | 4.06e-02 | 5.99e+01 | 74  |
| h0_1e12 | 1.0e12  | 1.92e-17 | 6.23e-04 | 3.90e+03 | 120 |
| h0_1e16 | 1.0e16  | 2.35e-14 | 9.28e-02 | 2.62e+01 | 120 |

PDF plot: [H0_scaling/no_riesz/qn_status_comparison.pdf](H0_scaling/no_riesz/qn_status_comparison.pdf)

## L-BFGS Storage Sweep (`lbfgs_storage`)

Fixed settings above this table:
- `storage_direct_*`:
  - `hcurl alpha1 = 1.0e5`
  - `hcurl alpha2 = 1.0e5`
- `storage_noscale_*`:
  - no metric norm scaling deck

| File name | Metric norm | Maximum Storage | Final objective | Final gnorm | gnorm0/gnormN | Iteration |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| storage_direct_mem05  | metric     | 5  | 1.22e-17 | 1.53e-09 | 3.96e+03 | 120 |
| storage_direct_mem10  | metric     | 10 | 7.64e-18 | 9.12e-10 | 6.66e+03 | 120 |
| storage_direct_mem20  | metric     | 20 | 5.28e-18 | 4.80e-10 | 1.26e+04 | 120 |
| storage_direct_mem40  | metric     | 40 | 5.17e-18 | 7.58e-10 | 8.00e+03 | 112 |
| storage_noscale_mem05 | no_scaling | 5  | 4.98e-18 | 3.45e-04 | 7.06e+03 | 120 |
| storage_noscale_mem10 | no_scaling | 10 | 4.42e-18 | 2.45e-04 | 9.92e+03 | 120 |
| storage_noscale_mem20 | no_scaling | 20 | 4.32e-18 | 4.16e-04 | 5.85e+03 | 120 |
| storage_noscale_mem40 | no_scaling | 40 | 3.69e-18 | 2.62e-04 | 9.29e+03 | 120 |

PDF plot: [lbfgs_storage/qn_status_comparison.pdf](lbfgs_storage/qn_status_comparison.pdf)

