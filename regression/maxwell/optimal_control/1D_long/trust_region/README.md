# trust_region

Trust-region (truncated CG) optimization on a single fixed mesh. The
main experiment is an alpha-scan over the H(curl) Riesz preconditioner
`P = alpha1 * M + alpha2 * K`.

## Sub-studies

- [alpha_scale](alpha_scale/) - sweep (alpha1, alpha2) including
  `auto`, `no_scale`, and a range of fixed values. See
  [alpha_scale/summary.md](alpha_scale/summary.md) for the full
  results table.
- [saved_runs](saved_runs/) - archived `direct_simple` and
  `no_scaling` runs at two CG budgets (`kry20_iter10`, `kry200_iter1`).
