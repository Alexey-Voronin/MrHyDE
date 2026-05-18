# lbfgs_ht_scale results

Joint h+t refinement: NZ and number of time steps scale together by the
same factor across r1, r2, r3.

| tag | NZ  | nsteps | factor |
|-----|-----|--------|--------|
| r1  | 44  | 400    | 1x     |
| r2  | 88  | 800    | 2x     |
| r3  | 132 | 1200   | 3x     |

## QN convergence summary

Final L-BFGS state (last reported iteration) per run:

| run        | iters | final value | final gnorm | gnorm0/gnormN |
|------------|-------|-------------|-------------|---------------|
| r1_noscale | 500   | 1.92e-20    | 9.52e-06    | 1.8e+05       |
| r1_auto    | 427   | 4.04e-20    | 9.56e-10    | 1.0e+06       |
| r2_noscale | 500   | 1.98e-19    | 1.50e-05    | 6.0e+04       |
| r2_auto    | 223   | 5.05e-19    | 1.73e-08    | 3.1e+04       |
| r3_noscale | 80    | 2.06e-15    | 2.13e-03    | 3.1e+02       |
| r3_auto    | 38    | 6.29e-17    | 1.13e-07    | 3.6e+03       |

## Convergence plot

[![qn_status_comparison](logs/qn_status_comparison.pdf)](logs/qn_status_comparison.pdf)
