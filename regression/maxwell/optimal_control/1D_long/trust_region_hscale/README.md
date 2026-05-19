# Optimization Results Summary

Mesh refinement levels (`r`) — NX, NY, and NZ all refine together (full h-refinement):

- **`r1`**: NX = 2, NY = 2, NZ = 44
- **`r2`**: NX = 4, NY = 4, NZ = 88
- **`r3`**: NX = 8, NY = 8, NZ = 176

Norm types (`scale`):

- **`noscale`**: Euclidean l^2 inner product on the discrete control (no Riesz preconditioner).
- **`ab1`**: H(curl) Sobolev Riesz map with `alpha1 = alpha2 = 1` (mass and curl terms weighted equally).
- **`auto`**: H(curl) Sobolev Riesz map with `alpha1`, `alpha2` auto-scaled to put mass on the same magnitude as curl-curl stiffness.


## Table 2 — kry20 (iter = 10, final)

*Inner Krylov budget = 20, outer TR iters = 10.*

| run | scale   | init value | init gnorm | value | gnorm | snorm | delta | #fval | #grad | tr | flagCG |
|-----|---------|------------|------------|-------|-------|-------|-------|-------|-------|----|--------|
| r1  | ab1     | 1.23e-11 | 9.99e-04 | 2.13e-18 | 2.73e-08 | 1.36e-10 | 1.00e+03 | 11 | 11 | 0 | 1 |
| r1  | auto    | 1.23e-11 | 9.68e-04 | **1.47e-18** | 1.88e-08 | 1.29e-10 | 1.00e+03 | 11 | 11 | 0 | 1 |
| r1  | noscale | 1.23e-11 | 1.70e+00 | 2.44e-18 | 1.43e-04 | 3.31e-14 | 1.00e+03 | 11 | 11 | 0 | 1 |
| r2  | ab1     | 1.24e-11 | 8.59e-04 | 1.35e-16 | 2.19e-06 | 5.86e-09 | 1.00e+00 | 11 | 11 | 0 | 1 |
| r2  | auto    | 1.24e-11 | 6.97e-04 | 5.54e-14 | 5.92e-06 | 2.45e-09 | 2.45e-09 | 11 | 6  | 0 | 3 |
| r2  | noscale | 1.24e-11 | 1.25e+00 | **6.30e-17** | 1.33e-03 | 1.81e-13 | 1.00e+03 | 11 | 11 | 0 | 1 |
| r3  | ab1     | 1.24e-11 | 8.76e-04 |       |       |       |       |    |    |    |    |
| r3  | auto    | 1.24e-11 | 5.00e-04 | **2.68e-14** | 3.06e-06 | 1.12e-08 | 7.01e-10 | 11 | 4  | 2 | 3 |
| r3  | noscale | 1.24e-11 | 1.10e+00 | 2.96e-12 | 2.87e-01 | 4.35e-13 | 4.35e-13 | 11 | 8  | 0 | 3 |

> `r3 / ab1` still running.


## Table 2 — kry200 (iter = 1)

*Inner Krylov budget = 200, outer TR iters = 1.*

| run | scale   | init value | init gnorm | value | gnorm | snorm | delta | #fval | #grad | tr | iterCG | flagCG |
|-----|---------|------------|------------|-------|-------|-------|-------|-------|-------|----|--------|--------|
| r1  | ab1     | 1.23e-11 | 9.99e-04 |  |  | — |  |  |  | — | — | — |
| r1  | auto    | 1.23e-11 | 9.68e-04 | **5.80e-20** | 1.21e-08 | 5.71e-08 | 1.00e-01 | 2 | 2 | 0 | 200 | 1 |
| r1  | noscale | 1.23e-11 | 1.70e+00 | 3.54e-19 | 8.52e-05 | 3.28e-11 | 1.00e-01 | 2 | 2 | 0 | 200 | 1 |
| r2  | ab1     | 1.24e-11 | 8.59e-04 | **1.24e-11** | 8.59e-04 | 1.00e-01 | 6.25e-03 | 2 | 1 | 2 | 151 | 2 |
| r2  | auto    | 1.24e-11 | 6.97e-04 | **1.24e-11** | 6.97e-04 | 1.00e-01 | 6.25e-03 | 2 | 1 | 2 | **27**  | 2 |
| r2  | noscale | 1.24e-11 | 1.25e+00 | **1.24e-11** | 1.25e+00 | 1.00e-01 | 6.25e-03 | 2 | 1 | 2 | 42  | 2 |
| r3  | ab1     |       |       |       |       |       |       |   |   |   |     |    |
| r3  | auto    | 1.24e-11 | 5.00e-04 | **1.24e-11** | 5.00e-04 | 1.00e-01 | 6.25e-03 | 2 | 1 | 2 | **23**  | 3 |
| r3  | noscale | 1.24e-11 | 1.10e+00 | **1.24e-11** | 1.10e+00 | 1.00e-01 | 6.25e-03 | 2 | 1 | 2 | 29  | 2 |


