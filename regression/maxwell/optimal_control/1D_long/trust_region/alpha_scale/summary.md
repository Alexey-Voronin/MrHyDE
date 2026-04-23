# Alpha Scaling Convergence Results

Trust-Region (Truncated CG) optimizer with preconditioner `P = alpha1 * M + alpha2 * K`.

## Table 1: Truncated CG (iterCG <= 20)

All runs use at most 50 inner CG iterations per trust-region step.

| Log file | alpha1 (a) | alpha2 (b) | Iters | Final objective | Final gnorm | #fval | #grad |
|---|---|---|---|---|---|---|---|
| `mrhyde_auto.log` | 1.20e11 (auto) | 1 | 50 | 2.95e-20 | 8.13e-10 | 51 | 51 |
| `mrhyde_no_scale.log` | M only | -- | 50 | 3.65e-20 | 5.27e-06 | 51 | 51 |
| `mrhyde_alpha_a1e11_b1.log` | 1e11 | 1 | 50 | 3.76e-20 | 8.56e-10 | 51 | 51 |
| `mrhyde_alpha_a1e13_b1.log` | 1e13 | 1 | 50 | 6.02e-20 | 8.83e-10 | 51 | 51 |
| `mrhyde_alpha_a1e1_b1e-11.log` | 10 | 1e-11 | 50 | 1.10e-19 | 1.61e-03 | 51 | 51 |
| `mrhyde_alpha_a1_b0.log` | 1 | 0 | 50 | 1.40e-19 | 5.88e-03 | 51 | 51 |
| `mrhyde_alpha_a1e15_b1.log` | 1e15 | 1 | 50 | 2.41e-19 | 2.62e-10 | 51 | 51 |
| `mrhyde_alpha_a1_b1.log` | 1 | 1 | 50 | 1.15e-17 | 3.21e-07 | 51 | 50 |
| `mrhyde_alpha_a1e5_b1.log` | 1e5 | 1 | 21 | 4.66e-17 | 8.70e-07 | 22 | 21 |
| `mrhyde_alpha_a1_b1e5.log` | 1 | 1e5 | 40 | 2.13e-16 | 8.18e-09 | 41 | 28 |
| `mrhyde_alpha_a0_b1.log` | 0 | 1 | 12 | 1.24e-15 | NaN | 13 | 11 |
| `mrhyde_alpha_a1e5_b1e5.log` | 1e5 | 1e5 | -- | -- | -- | -- | -- |
| `mrhyde_alpha_a1e8_b1.log` | 1e8 | 1 | -- | -- | -- | -- | -- |

## Table 2: Full CG (iterCG <= 1000)

Same configurations but with up to 1000 inner CG iterations.

| Log file | alpha1 (a) | alpha2 (b) | Iters | Final objective | Final gnorm | #fval | #grad |
|---|---|---|---|---|---|---|---|
| `mrhyde_no_scale_full_cg.log` | P=I | -- | 1 | 7.62e-23 | 4.10e-07 | 2 | 2 |
| `mrhyde_auto_full_cg.log` | 1.20e11 (auto) | 1 | 1 | 3.05e-19 | 1.95e-08 | 2 | 2 |

