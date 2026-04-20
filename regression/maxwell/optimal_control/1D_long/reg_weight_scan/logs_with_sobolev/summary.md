# Regularization Weight Scan -- Auto-Scaled Sobolev Riesz Map

**Setup:** P = alpha1*M + alpha2*K with auto-scaling (alpha1 = mean(diag(K))/mean(diag(M)) ~ 1e11, alpha2 = 1).
w_obj = 1e35, L-BFGS mem 20, 200 iters, 2x2x44 hex mesh.

## Convergence Data

```text
file                                   s1           s2           s3           s4           s5           s6           s7           s8
mrhyde_l2_0_curl_1e11.log         1.1e-16      1.8e-17      6.3e-18      3.1e-18      2.0e-18      1.3e-18      1.0e-18      6.9e-19
mrhyde_l2_1e11_curl_0.log         4.1e-16      4.2e-17      1.5e-17      7.1e-18      4.3e-18      2.7e-18      2.0e-18      1.3e-18
mrhyde_l2_1e11_curl_1.log         4.1e-16      4.2e-17      1.5e-17      7.1e-18      4.3e-18      2.7e-18      2.0e-18      1.3e-18
mrhyde_l2_1e16_curl_1e5.log       4.1e-16      4.2e-17      1.5e-17      7.1e-18      4.3e-18      2.7e-18      2.0e-18      1.3e-18
mrhyde_l2_0_curl_1e5.log          4.1e-16      4.2e-17      1.5e-17      7.1e-18      4.3e-18      2.7e-18      2.0e-18      1.3e-18
mrhyde_l2_1e13_curl_1e5.log       4.1e-16      4.2e-17      1.5e-17      7.1e-18      4.3e-18      2.7e-18      2.0e-18      1.3e-18
mrhyde_l2_1e5_curl_0.log          4.1e-16      4.2e-17      1.5e-17      7.1e-18      4.3e-18      2.7e-18      2.0e-18      1.3e-18
mrhyde_l2_1e5_curl_1e5.log        4.1e-16      4.2e-17      1.5e-17      7.1e-18      4.3e-18      2.7e-18      2.0e-18      1.3e-18
mrhyde_l2_1e8_curl_1e5.log        4.1e-16      4.2e-17      1.5e-17      7.1e-18      4.3e-18      2.7e-18      2.0e-18      1.3e-18
mrhyde_l2_1e20_curl_1e9.log       4.1e-16      4.2e-17      1.5e-17      7.2e-18      4.3e-18      2.7e-18      2.0e-18      1.3e-18
mrhyde_l2_1e25_curl_1e14.log      2.0e-16      1.1e-16      9.9e-17      9.6e-17      9.5e-17      9.4e-17      9.4e-17      9.4e-17
mrhyde_l2_1e5_curl_1e16.log       5.4e-14      1.0e-14      8.9e-15      8.5e-15      8.4e-15      8.3e-15      8.3e-15      8.3e-15
mrhyde_l2_1e30_curl_1e19.log      1.2e-11      1.2e-11      6.0e-12      6.0e-12      6.0e-12      5.5e-12      5.5e-12      5.5e-12
mrhyde_l2_1e34_curl_1e23.log      1.2e-11      1.2e-11      1.2e-11      1.2e-11      1.2e-11      1.2e-11      1.2e-11      1.2e-11
Parsed datasets: 14
```

## Effective Regularization Magnitudes

- Eff. L2 = w_l2
- Eff. curl = w_curl

Same visibility threshold as P=I: max(eff_L2, eff_curl) > w_obj * J_tracking.

## Four Behavioral Groups

| Grp | Final obj | Mechanism |
|---|---|---|
| 1 | ~1.3e-18 | Reg invisible. 9 runs identical. Sobolev improves over P=I (1.3e-18 vs 1.1e-18 at s8, but faster early convergence). |
| 1* | 6.9e-19 | curl_1e11 is the best run overall -- curl reg at visibility threshold seems to help. |
| 2 | ~9.4e-17 | Reg-tracking equilibrium. Better than P=I plateau (9.4e-17 vs 4.4e-16, ~5x). |
| 3 | ~8.3e-15 | Curl-dominated reg restricts feasible controls. Slightly better than P=I (8.3e-15 vs 1.1e-14). |
| 4 | ~1e-11 | Reg overwhelms tracking. Same as P=I -- Sobolev cannot help when reg dominates. |


## Convergence Plot

![Quasi-Newton Status Comparison](qn_status_comparison.pdf)
