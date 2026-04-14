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

| File name | Metric norm | `alpha1` | `alpha2` | Final objective | Iteration |
| --- | --- | ---: | ---: | ---: | ---: |
| no_scaling | no_scaling | N/A | N/A | 4.320354e-18 | 120 |
| s1e3_r1em2 | metric | 1.0e3 | 1.0e1 | 3.466666e-17 | 51 |
| s1e3_r1e0 | metric | 1.0e3 | 1.0e3 | 8.763888e-18 | 109 |
| s1e3_r1e2 | metric | 1.0e3 | 1.0e5 | 5.126518e-18 | 120 |
| s1e5_r1em2 | metric | 1.0e5 | 1.0e3 | 6.241212e-18 | 120 |
| s1e5_r1e0 | metric | 1.0e5 | 1.0e5 | 5.280814e-18 | 120 |
| s1e5_r1e2 | metric | 1.0e5 | 1.0e7 | 1.195091e-17 | 94 |
| s1e7_r1em2 | metric | 1.0e7 | 1.0e5 | 2.316429e-17 | 68 |
| s1e7_r1e0 | metric | 1.0e7 | 1.0e7 | 7.294439e-18 | 120 |
| s1e7_r1e2 | metric | 1.0e7 | 1.0e9 | 7.848285e-18 | 120 |

PDF plot: [alpha_ratio/qn_status_comparison.pdf](alpha_ratio/qn_status_comparison.pdf)


`alpha2=0` performs onlt slightly worse than `no_scaling` case in the beginning, but at around 80 iterations it beginds to platau. 

## H0 Scaling Sweep (`H0_scaling`)

Fixed settings above this table:
- Metric norm: `metric` (`direct_simple_alpha_1e5` family)
- `hcurl alpha1 = 1.0e5`
- `hcurl alpha2 = 1.0e5`

| File name | Metric norm | H0 scale | Final objective | Iteration |
| --- | --- | ---: | ---: | ---: |
| h0scale_1em2 | metric | 1.0e-2 | 9.959054e-18 | 106 |
| h0scale_1em1 | metric | 1.0e-1 | 4.876338e-18 | 120 |
| h0scale_1e0 | metric | 1.0e0 | 4.729855e-17 | 111 |
| h0scale_1e1 | metric | 1.0e1 | 5.213664e-16 | 120 |
| h0scale_1e2 | metric | 1.0e2 | 5.930977e-15 | 61 |

PDF plot: [H0_scaling/qn_status_comparison.pdf](H0_scaling/qn_status_comparison.pdf)

## L-BFGS Storage Sweep (`lbfgs_storage`)

Fixed settings above this table:
- `storage_direct_*`:
  - `hcurl alpha1 = 1.0e5`
  - `hcurl alpha2 = 1.0e5`
- `storage_noscale_*`:
  - no metric norm scaling deck

| File name | Metric norm | Maximum Storage | Final objective | Iteration |
| --- | --- | ---: | ---: | ---: |
| storage_direct_mem05 | metric | 5 | 1.215605e-17 | 120 |
| storage_direct_mem10 | metric | 10 | 7.636979e-18 | 120 |
| storage_direct_mem20 | metric | 20 | 5.280814e-18 | 120 |
| storage_direct_mem40 | metric | 40 | 5.170400e-18 | 112 |
| storage_noscale_mem05 | no_scaling | 5 | 4.984076e-18 | 120 |
| storage_noscale_mem10 | no_scaling | 10 | 4.422435e-18 | 120 |
| storage_noscale_mem20 | no_scaling | 20 | 4.320354e-18 | 120 |
| storage_noscale_mem40 | no_scaling | 40 | 3.693500e-18 | 120 |

PDF plot: [lbfgs_storage/qn_status_comparison.pdf](lbfgs_storage/qn_status_comparison.pdf)

