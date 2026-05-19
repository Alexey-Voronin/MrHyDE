# lbfgs_t_scale results

3x3 sweep: mesh (NZ) and number of time steps vary independently. The same
three time levels (t0/t1/t2 = 400/800/1600) are run on each of the three
meshes (r1/r2/r3, matching the NZ values in `lbfgs_ht_scale`).

| tag    | NZ  | nsteps | mesh factor | t factor |
|--------|-----|--------|-------------|----------|
| r1_t0  | 44  | 400    | 1x          | 1x       |
| r1_t1  | 44  | 800    | 1x          | 2x       |
| r1_t2  | 44  | 1600   | 1x          | 4x       |
| r2_t0  | 88  | 400    | 2x          | 1x       |
| r2_t1  | 88  | 800    | 2x          | 2x       |
| r2_t2  | 88  | 1600   | 2x          | 4x       |
| r3_t0  | 132 | 400    | 3x          | 1x       |
| r3_t1  | 132 | 800    | 3x          | 2x       |
| r3_t2  | 132 | 1600   | 3x          | 4x       |

## QN convergence summary

Final L-BFGS state (last reported iteration) per run:

| run           | iters | final value | final gnorm | gnorm0/gnormN |
|---------------|-------|-------------|-------------|---------------|
| r1_t0_noscale | TBD   | TBD         | TBD         | TBD           |
| r1_t0_auto    | TBD   | TBD         | TBD         | TBD           |
| r1_t1_noscale | TBD   | TBD         | TBD         | TBD           |
| r1_t1_auto    | TBD   | TBD         | TBD         | TBD           |
| r1_t2_noscale | TBD   | TBD         | TBD         | TBD           |
| r1_t2_auto    | TBD   | TBD         | TBD         | TBD           |
| r2_t0_noscale | TBD   | TBD         | TBD         | TBD           |
| r2_t0_auto    | TBD   | TBD         | TBD         | TBD           |
| r2_t1_noscale | TBD   | TBD         | TBD         | TBD           |
| r2_t1_auto    | TBD   | TBD         | TBD         | TBD           |
| r2_t2_noscale | TBD   | TBD         | TBD         | TBD           |
| r2_t2_auto    | TBD   | TBD         | TBD         | TBD           |
| r3_t0_noscale | TBD   | TBD         | TBD         | TBD           |
| r3_t0_auto    | TBD   | TBD         | TBD         | TBD           |
| r3_t1_noscale | TBD   | TBD         | TBD         | TBD           |
| r3_t1_auto    | TBD   | TBD         | TBD         | TBD           |
| r3_t2_noscale | TBD   | TBD         | TBD         | TBD           |
| r3_t2_auto    | TBD   | TBD         | TBD         | TBD           |

## Convergence plot

[![qn_status_comparison](logs/qn_status_comparison.pdf)](logs/qn_status_comparison.pdf)
