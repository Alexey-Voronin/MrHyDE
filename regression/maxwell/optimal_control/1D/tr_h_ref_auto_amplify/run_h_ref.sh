#!/bin/bash
#
# Trust-Region sibling of lbfgs_h_ref: same h-only refinement
# (NX=NY=2, NZ in {44,88,132}, nsteps=400 fixed), multiplicative
# EM-Energy objective. Two Riesz modes (noscale, auto). Step block
# uses Trust Region with a truncated-CG subproblem solver and FD
# HessVec (no secant memory).
#
# Per-run cost: ~50 TR outer iter * up to 20 Krylov inner = ~1000
# fwd+adj pairs. ~10x the lbfgs_h_ref wall-time.

source ../../../../../../scripts/load-env.sh

set -e
cd "$(dirname "$0")"

TAGS="r1 r2 r3"
MODES="noscale"
MRHYDE="mrhyde"

np_for_tag() {
  case "$1" in
    r1|r2|r3) echo 22 ;;
    *) echo "unknown tag: $1" >&2; exit 1 ;;
  esac
}

for tag in $TAGS; do
  for mode in $MODES; do
    runtag="${tag}_${mode}"
    logfile="mrhyde_${runtag}.log"
    inputfile="input_${runtag}.yaml"
    roldeck="rol_${runtag}.yaml"
    csvfile="mrhyde_riesz_diag_${runtag}.csv"
    nsteps=400

    sed -e "s|RIESZ_CSV_PLACEHOLDER|${csvfile}|" \
        "rol_decks/rol_${mode}.yaml" > "rol_decks/${roldeck}"

    sed -e "s|MESH_FILE_PLACEHOLDER|meshes/mesh_${tag}.yaml|" \
        -e "s|ROL_FILE_PLACEHOLDER|rol_decks/${roldeck}|" \
        -e "s|NSTEPS_PLACEHOLDER|${nsteps}|" \
        input_base.yaml > "$inputfile"

    echo "=== Running tag=${tag}, mode=${mode}, nsteps=${nsteps} ==="
    echo "  Input: ${inputfile}"
    echo "  Log:   ${logfile}"
    echo "  CSV:   ${csvfile}"

    np=$(np_for_tag "$tag")
    mpiexec -n "$np" "$MRHYDE" "$inputfile" >& "logs/${logfile}" || {
      echo "  FAILED (exit code $?), see logs/${logfile}"
      rm -f "$inputfile" "rol_decks/${roldeck}"
      continue
    }
    [ -f "$csvfile" ] && mv "$csvfile" "logs/${csvfile}"

    echo "  --- Diagnostics ---"
    grep -E "RieszAutoScale|RieszMap|hcurl alpha" "logs/${logfile}" | head -4
    echo "  --- TR convergence (first 12 lines) ---"
    sed -n '/^  iter  value/,/^$/p' "logs/${logfile}" | head -12
    echo ""

    rm -f "$inputfile" "rol_decks/${roldeck}"
  done
done

echo "=== All runs complete ==="
echo "Log files:"
ls -la logs/mrhyde_r*.log 2>/dev/null
