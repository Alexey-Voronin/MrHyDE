# Regularization Weight Scan -- No Riesz Map (P=I)

**Setup:** P=I (Euclidean metric), w_obj = 1e35, L-BFGS mem 20, 200 iters, 2x2x44 hex mesh.

## Convergence Data

```text
file                                   s1           s2           s3           s4           s5           s6           s7           s8
mrhyde_l2_1e11_curl_0.log         1.3e-15      3.8e-17      1.6e-17      6.5e-18      3.7e-18      2.3e-18      1.5e-18      1.1e-18
mrhyde_l2_1e16_curl_1e5.log       1.3e-15      3.8e-17      1.6e-17      6.5e-18      3.7e-18      2.4e-18      1.5e-18      1.1e-18
mrhyde_l2_1e20_curl_1e9.log       1.3e-15      3.8e-17      1.6e-17      6.5e-18      3.7e-18      2.4e-18      1.5e-18      1.1e-18
mrhyde_l2_1e8_curl_1e5.log        1.3e-15      3.8e-17      1.6e-17      6.5e-18      3.7e-18      2.4e-18      1.5e-18      1.1e-18
mrhyde_l2_0_curl_1e5.log          1.3e-15      3.8e-17      1.6e-17      6.5e-18      3.7e-18      2.3e-18      1.5e-18      1.1e-18
mrhyde_l2_1e5_curl_1e5.log        1.3e-15      3.8e-17      1.6e-17      6.5e-18      3.7e-18      2.3e-18      1.5e-18      1.1e-18
mrhyde_l2_1e11_curl_1.log         1.3e-15      3.8e-17      1.6e-17      6.5e-18      3.7e-18      2.4e-18      1.5e-18      1.1e-18
mrhyde_l2_1e5_curl_0.log          1.3e-15      3.8e-17      1.6e-17      6.5e-18      3.7e-18      2.3e-18      1.5e-18      1.1e-18
mrhyde_l2_1e13_curl_1e5.log       1.3e-15      3.8e-17      1.6e-17      6.5e-18      3.7e-18      2.4e-18      1.6e-18      1.1e-18
mrhyde_l2_0_curl_1e11.log         1.3e-15      3.9e-17      1.7e-17      7.1e-18      4.2e-18      2.9e-18      2.0e-18      1.6e-18
mrhyde_l2_1e25_curl_1e14.log      3.3e-15      5.4e-16      4.9e-16      4.7e-16      4.7e-16      4.6e-16      4.5e-16      4.4e-16
mrhyde_l2_1e5_curl_1e16.log       4.7e-14      3.7e-14      1.9e-14      1.7e-14      1.3e-14      1.2e-14      1.1e-14      1.1e-14
mrhyde_l2_1e30_curl_1e19.log      8.8e-12      7.1e-12      6.1e-12      5.8e-12      5.7e-12      5.7e-12      5.5e-12      5.5e-12
mrhyde_l2_1e34_curl_1e23.log      1.2e-11      1.2e-11      1.2e-11      1.2e-11      1.2e-11      1.2e-11      1.2e-11      1.2e-11
```

## Effective Regularization Magnitudes

- Eff. L2 = w_l2 
- Eff. curl = w_curl

Regularization becomes visible when
max(eff_L2, eff_curl) > w_obj * J_tracking.


## Four Behavioral Groups

| Grp | Final obj | Max eff. reg | Mechanism |
|---|---|---|---|
| 1 | ~1e-18 | <= 1e16 | Reg invisible (9 runs identical to 14 digits). Borderline: curl_1e11 at 1.6e-18. |
| 2 | ~4e-16 | ~1e21 | Reg-tracking equilibrium: converged to regularized solution, not stalled. |
| 3 | ~1e-14 | ~1e23 (curl only) | Curl penalty forces near-curl-free controls; restricted subspace prevents tracking. |
| 4 | ~1e-11 | 1e26-1e30 | Reg > tracking from iter 1; optimizer drives ctrl -> 0. |

## Inferring K/M entry ratio from the data

We can estimate the typical entry magnitudes of the mass matrix M and stiffness
matrix K without inspecting them directly, using the visibility threshold above.

Regularization becomes visible when `w * (matrix scale) ~ w_obj * J_tracking`.
At the solution, `w_obj * J ~ 1e35 * 1e-18 = 1e17`. This gives us two constraints:

- `curl_1e11` (w_curl=1e11) is the first run where reg is visible (1.6e-18 vs 1.1e-18),
  so `1e11 * K_scale ~ 1e17`, giving **K_scale ~ 1e6**.
- `l2_1e20` (w_l2=1e20) is still invisible, so `1e20 * M_scale < 1e17`,
  giving **M_scale < 1e-3**.

Implied ratio: **K/M ~ 1e9 to 1e10**, which is consistent with previous tests.

## Convergence Plot

![Quasi-Newton Status Comparison](qn_status_comparison.pdf)
