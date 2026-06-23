# lbfgs_grad_dump

Minimal reproducer for iter-0 gradient differences between two objective spellings:

- pow form: `(E[x])^2 + (E[y])^2 + (E[z])^2`
- mul form: `E[x]*E[x] + E[y]*E[y] + E[z]*E[z]`

This version runs an `NZ x N_t` matrix at fixed `np=6`:

- `NZ in {44, 88, 132, 176}`
- `N_t in {400, 800, 1200, 1600}`

## Run

```bash
./run_dump.sh
python3 analyze_full.py > analysis_full.log
```
