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
