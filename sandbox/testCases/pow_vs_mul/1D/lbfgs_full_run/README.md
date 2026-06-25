# lbfgs_noscale_pow_test

Minimal 1D Maxwell optimal-control baseline against a freshly-cloned
upstream MrHyDE.

## Two decks

- `input_base_pow.yaml` -- objective uses `(E[x])^2` etc., which
  routes through `pow(base, 2.0)` in the function manager.
- `input_base_mul.yaml` -- objective uses `E[x]*E[x]` etc., which
  routes through the multiplication operator.

Everything else is identical between the two: mesh, nsteps, ROL
deck, tolerances.

## Running

```
./run_ht_scale.sh
```
