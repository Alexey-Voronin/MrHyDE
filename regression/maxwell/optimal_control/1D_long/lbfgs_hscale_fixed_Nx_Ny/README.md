# lbfgs_hscale_fixed_Nx_Ny results

h-only refinement on the long z-axis (NX = NY = 2 fixed, NZ varies);
time stepping is held fixed (`final time: 2.0e-13`, `number of steps:
400`).

| tag | NX | NY | NZ  | nsteps | factor |
|-----|----|----|-----|--------|--------|
| r1  | 2  | 2  | 44  | 400    | 1x     |
| r2  | 2  | 2  | 88  | 400    | 2x     |
| r3  | 2  | 2  | 132 | 400    | 3x     |

## QN convergence summary

Final L-BFGS state (last reported iteration) per run:

| run        | iters | final value | final gnorm | gnorm0/gnormN |
|------------|-------|-------------|-------------|---------------|
| r1_noscale | 500   | 1.92e-20    | 9.52e-06    | 1.8e+05       |
| r1_auto    | 427   | 4.04e-20    | 9.56e-10    | 1.0e+06       |
| r1_ab1     | 500   | 5.40e-20    | 1.08e-08    | 9.2e+04       |
| r2_noscale | 500   | 1.93e-20    | 1.11e-05    | 1.1e+05       |
| r2_auto    | 130   | 1.45e-18    | 4.95e-08    | 1.5e+04       |
| r2_ab1     | 61    | 2.47e-17    | 2.62e-07    | 3.3e+03       |
| r3_noscale | 37    | 3.29e-15    | 1.37e-02    | 8.3e+01       |
| r3_auto    | 47    | 4.18e-17    | 2.68e-07    | 2.6e+03       |
| r3_ab1     | 36    | 1.17e-16    | 7.88e-07    | 1.1e+03       |

## Convergence plot

[![qn_status_comparison](logs/qn_status_comparison.pdf)](logs/qn_status_comparison.pdf)
