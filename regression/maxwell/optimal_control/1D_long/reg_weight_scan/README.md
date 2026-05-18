# Regularization Weight Scan Results -- With Sobolev Riesz Map

**Setup:** 1D Maxwell optimal control, L-BFGS (memory 20), **auto-scaled Sobolev Riesz map** (`parameter gradient preconditioner type: sparse_direct`, alpha1 = mean(diag(K))/mean(diag(M)) ~ 1e11, alpha2 = 1).
Tracking weight: `w_obj = 1e35`. Max iterations: 200.
Mesh: 2x2x44 hex, hz ~ 1.82 um. K/M entry ratio ~ 1e11.

## Results Table

Sorted by final objective value (ascending).

| Label | w_l2 | w_curl | w_l2/w_curl | Final obj | Final iter | Termination |
|---|---|---|---|---|---|---|
| l2_0_curl_1e11 | 0 | 1e11 | 0 (curl only) | 6.89e-19 | 200 | Iter limit |
| l2_0_curl_1e5 | 0 | 1e5 | 0 (curl only) | 1.30e-18 | 141 | Step tol |
| l2_1e5_curl_0 | 1e5 | 0 | inf (L2 only) | 1.30e-18 | 141 | Step tol |
| l2_1e5_curl_1e5 | 1e5 | 1e5 | 1 | 1.30e-18 | 141 | Step tol |
| l2_1e11_curl_0 | 1e11 | 0 | inf (L2 only) | 1.30e-18 | 141 | Step tol |
| l2_1e11_curl_1 | 1e11 | 1 | 1e11 | 1.30e-18 | 141 | Step tol |
| l2_1e8_curl_1e5 | 1e8 | 1e5 | 1e3 | 1.30e-18 | 141 | Step tol |
| l2_1e13_curl_1e5 | 1e13 | 1e5 | 1e8 | 1.30e-18 | 141 | Step tol |
| l2_1e16_curl_1e5 | 1e16 | 1e5 | 1e11 | 1.30e-18 | 141 | Step tol |
| l2_1e20_curl_1e9 | 1e20 | 1e9 | 1e11 | 1.30e-18 | 141 | Step tol |
| l2_1e25_curl_1e14 | 1e25 | 1e14 | 1e11 | 9.40e-17 | 200 | Step tol |
| l2_1e5_curl_1e16 | 1e5 | 1e16 | 1e-11 (curl heavy) | 8.32e-15 | 37 | Step tol |
| l2_1e30_curl_1e19 | 1e30 | 1e19 | 1e11 | 5.55e-12 | 3 | Step tol |
| l2_1e34_curl_1e23 | 1e34 | 1e23 | 1e11 | 1.23e-11 | 2 | Step tol |

## Convergence Snapshots (objective value at selected iterations)

| Label | w_l2 | w_curl | iter 1 | iter 5 | iter 10 | iter 20 | iter 50 | iter 100 | iter 141 | iter 200 |
|---|---|---|---|---|---|---|---|---|---|---|
| l2_0_curl_1e11 | 0 | 1e11 | 1.09e-12 | 1.94e-14 | 1.67e-15 | 2.28e-16 | 1.75e-17 | 3.07e-18 | 1.38e-18 | 6.89e-19 |
| l2_0_curl_1e5 | 0 | 1e5 | 1.09e-12 | 1.94e-14 | 1.67e-15 | 2.27e-16 | 1.75e-17 | 2.99e-18 | 1.30e-18 | -- |
| l2_1e5_curl_0 | 1e5 | 0 | 1.09e-12 | 1.94e-14 | 1.67e-15 | 2.27e-16 | 1.75e-17 | 2.99e-18 | 1.30e-18 | -- |
| l2_1e5_curl_1e5 | 1e5 | 1e5 | 1.09e-12 | 1.94e-14 | 1.67e-15 | 2.27e-16 | 1.75e-17 | 2.99e-18 | 1.30e-18 | -- |
| l2_1e11_curl_0 | 1e11 | 0 | 1.09e-12 | 1.94e-14 | 1.67e-15 | 2.27e-16 | 1.75e-17 | 2.99e-18 | 1.30e-18 | -- |
| l2_1e11_curl_1 | 1e11 | 1 | 1.09e-12 | 1.94e-14 | 1.67e-15 | 2.27e-16 | 1.75e-17 | 2.99e-18 | 1.30e-18 | -- |
| l2_1e8_curl_1e5 | 1e8 | 1e5 | 1.09e-12 | 1.94e-14 | 1.67e-15 | 2.27e-16 | 1.75e-17 | 2.99e-18 | 1.30e-18 | -- |
| l2_1e13_curl_1e5 | 1e13 | 1e5 | 1.09e-12 | 1.94e-14 | 1.67e-15 | 2.27e-16 | 1.75e-17 | 2.99e-18 | 1.30e-18 | -- |
| l2_1e16_curl_1e5 | 1e16 | 1e5 | 1.09e-12 | 1.94e-14 | 1.67e-15 | 2.27e-16 | 1.75e-17 | 2.99e-18 | 1.30e-18 | -- |
| l2_1e20_curl_1e9 | 1e20 | 1e9 | 1.09e-12 | 1.94e-14 | 1.67e-15 | 2.27e-16 | 1.75e-17 | 2.99e-18 | 1.30e-18 | -- |
| l2_1e25_curl_1e14 | 1e25 | 1e14 | 1.09e-12 | 1.95e-14 | 1.76e-15 | 3.20e-16 | 1.10e-16 | 9.60e-17 | 9.46e-17 | 9.40e-17 |
| l2_1e5_curl_1e16 | 1e5 | 1e16 | 1.09e-12 | 2.74e-14 | 9.81e-15 | 8.45e-15 | -- | -- | -- | -- |
| l2_1e30_curl_1e19 | 1e30 | 1e19 | 5.96e-12 | -- | -- | -- | -- | -- | -- | -- |
| l2_1e34_curl_1e23 | 1e34 | 1e23 | 1.23e-11 | -- | -- | -- | -- | -- | -- | -- |

## Comparison: Sobolev Riesz Map vs P=I (from logs_orig)

At iteration 120 (the P=I iteration limit):

| Label | w_l2 | w_curl | P=I (iter 120) | Sobolev (iter 120) | Speedup |
|---|---|---|---|---|---|
| l2_1e5_curl_1e5 | 1e5 | 1e5 | 4.32e-18 | 1.75e-18 | 2.5x |
| l2_1e5_curl_0 | 1e5 | 0 | 4.32e-18 | 1.75e-18 | 2.5x |
| l2_0_curl_1e5 | 0 | 1e5 | 4.32e-18 | 1.75e-18 | 2.5x |
| l2_1e25_curl_1e14 | 1e25 | 1e14 | 4.60e-16 | 9.49e-17 | 4.8x |
| l2_1e5_curl_1e16 | 1e5 | 1e16 | 1.25e-14 | 8.32e-15 | 1.5x |
| l2_1e30_curl_1e19 | 1e30 | 1e19 | 5.50e-12 (stalled@18) | 5.55e-12 (stalled@3) | none |
| l2_1e34_curl_1e23 | 1e34 | 1e23 | 1.23e-11 (stalled@20) | 1.23e-11 (stalled@2) | none |

## Key Observations

1. **Sobolev Riesz map converges deeper than P=I.** The main cluster of runs
   (w_l2 <= 1e20) converges to 1.30e-18 and hits step tolerance at iteration 141,
   vs P=I which reached only 4.32e-18 at the 120-iteration limit. This is a 3.3x
   improvement in final objective.

2. **One outlier goes further.** `l2_0_curl_1e11` (curl-only, w_curl=1e11) did not
   stall at step tolerance and reached 6.89e-19 at iteration 200 -- the best result
   in the entire scan. This is the only run where curl regularization weight is large
   enough to contribute meaningfully to the gradient while still small enough not to
   degrade tracking.

3. **Regularization weights up to ~1e16 remain invisible**, same as the P=I scan.
   All runs with w_l2 <= 1e16 and w_curl <= 1e5 produce identical convergence
   histories. The Sobolev Riesz map does not change this: it rescales the gradient,
   but if the regularization contribution to the gradient is negligible, there is
   nothing to rescale.

4. **The Sobolev Riesz map does NOT compensate for regularization weight imbalance.**
   The `l2_1e5_curl_1e16` (curl-heavy) run still degrades (8.32e-15 vs 1.30e-18),
   and the heavy-reg runs (1e30, 1e34) still stall immediately. The Riesz map helps
   the solver converge faster on the same problem, but does not fix a poorly posed
   objective.

5. **Sobolev converges ~2.5x better in the tracking-dominated regime.** For all runs
   where regularization is invisible, the improvement at iter 120 is consistently
   about 2.5x (4.32e-18 -> 1.75e-18). The benefit is larger (~4.8x) when
   regularization is active but not overwhelming (l2_1e25_curl_1e14).

6. **Heavy regularization stalls faster with Sobolev.** The 1e30 and 1e34 runs stall
   at iteration 3 and 2 respectively (vs 18 and 20 with P=I). The Riesz map amplifies
   the regularization-dominated gradient more efficiently, so the optimizer recognizes
   the stall sooner.

## Plots

- [Sobolev Riesz map convergence](logs_with_sobolev/qn_status_comparison.pdf)
- [P=I (no Riesz map) convergence](logs_l2/qn_status_comparison.pdf)
