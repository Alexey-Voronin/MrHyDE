# Optimization Results Summary

Mesh refinement levels (`r`) — only NZ varies along the long z-axis:

- **`r1`**: NX = 2, NY = 2, NZ = 44
- **`r2`**: NX = 2, NY = 2, NZ = 88
- **`r3`**: NX = 2, NY = 2, NZ = 132

Norm types (`scale`):

- **`noscale`**: Euclidean l^2 inner product on the discrete control (no Riesz preconditioner).
- **`ab1`**: H(curl) Sobolev Riesz map with `alpha1 = alpha2 = 1` (mass and curl terms weighted equally).
- **`auto`**: H(curl) Sobolev Riesz map with `alpha1`, `alpha2` auto-scaled to put mass on the same magnitude as curl-curl stiffness.


## Table 2 — kry20 (iter = 10, final)

*Inner Krylov budget = 20, outer TR iters = 10.*     

| run | scale   | value | gnorm | snorm | delta | #fval | #grad | tr | flagCG |
|-----|---------|-------|-------|-------|-------|-------|-------|----|--------|
| r1  | ab1     | 2.13e-18 | 2.73e-08 | 1.36e-10 | 1.00e+03 | 11 | 11 | 0 | 1 |
| r1  | auto    | **1.47e-18** | 1.88e-08 | 1.29e-10 | 1.00e+03 | 11 | 11 | 0 | 1 |
| r1  | noscale | 2.44e-18 | 1.43e-04 | 3.31e-14 | 1.00e+03 | 11 | 11 | 0 | 1 |
| r2  | ab1     | **8.20e-16** | 5.17e-06 | 1.53e-05 | 9.54e-07 | 11 | 6  | 2 | 2 |
| r2  | auto    | 4.35e-14 | 2.08e-05 | 2.40e-08 | 2.40e-08 | 11 | 8  | 0 | 3 |
| r2  | noscale | 1.02e-13 | 1.11e-01 | 1.88e-12 | 1.88e-12 | 11 | 8  | 0 | 3 |
| r3  | ab1     | 8.22e-13 | 5.73e-05 | 2.43e-09 | 2.43e-09 | 11 | 7  | 0 | 3 |
| r3  | auto    | **5.12e-13** | 5.58e-05 | 4.94e-08 | 7.35e-09 | 11 | 7  | 2 | 3 |
| r3  | noscale | 3.48e-12 | 5.19e-01 | 2.05e-11 | 2.05e-11 | 11 | 2  | 0 | 3 |


## Table 2 — kry200 (iter = 1)

+*Inner Krylov budget = 200, outer TR iters = 1.* 

| run | scale   | value | gnorm | snorm | delta | #fval | #grad | tr | iterCG | flagCG |
|-----|---------|-------|-------|-------|-------|-------|-------|----|--------|--------|
| r1  | ab1     | 1.16e-19 | 8.63e-09 | 5.53e-08 | 1.00e-01 | 2 | 2 | 0 | 200 | 1 |
| r1  | auto    | **5.80e-20** | 1.21e-08 | 5.71e-08 | 1.00e-01 | 2 | 2 | 0 | 200 | 1 |
| r1  | noscale | 3.54e-19 | 8.52e-05 | 3.28e-11 | 1.00e-01 | 2 | 2 | 0 | 200 | 1 |
| r2  | ab1     | **1.24e-11** | 8.59e-04 | 1.00e-01 | 6.25e-03 | 2 | 1 | 2 | 112 | 2 |
| r2  | auto    | 1.24e-11 | 7.49e-04 | 1.00e-01 | 6.25e-03 | 2 | 1 | 2 | 130 | 2 |
| r2  | noscale | 1.24e-11 | 1.25e+00 | 6.20e-08 | 1.55e-08 | 2 | 1 | 2 | 200 | 1 |
| r3  | ab1     | **1.24e-11** | 8.64e-04 | 1.00e-01 | 6.25e-03 | 2 | 1 | 2 | 50  | 2 |
| r3  | auto    | 1.24e-11 | 6.84e-04 | 1.00e-01 | 6.25e-03 | 2 | 1 | 2 | 29  | 2 |
| r3  | noscale | 1.24e-11 | 1.14e+00 | 1.00e-01 | 6.25e-03 | 2 | 1 | 2 | 18  | 2 |
