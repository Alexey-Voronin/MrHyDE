#!/usr/bin/env bash
#
# Scan over regularization weights (w_l2, w_curl) in the objective function.
# No Sobolev Riesz map (P=I) -- isolates the effect of objective weights.
#
# Each run uses a fixed input_rol.yaml (L-BFGS, no Riesz map) and generates
# a per-run input.yaml with the specified w_l2 and w_curl values.
#
# Usage: bash run_scan.sh [--np 8]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR" || exit 1

# --- MrHyDE binary -----------------------------------------------------------
MRHYDE_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd -P)"
MRHYDE_BIN="$MRHYDE_ROOT/mrhyde.exe"
MRHYDE_BIN_RESOLVED="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$MRHYDE_BIN")"

if [[ ! -x "$MRHYDE_BIN" ]]; then
  echo "error: mrhyde is missing or not executable: $MRHYDE_BIN" >&2
  exit 1
fi
if [[ ! -x "$MRHYDE_BIN_RESOLVED" ]]; then
  echo "error: resolved mrhyde target is missing or not executable: $MRHYDE_BIN_RESOLVED" >&2
  exit 1
fi
echo "Using mrhyde link:   $MRHYDE_BIN"
echo "Using mrhyde target: $MRHYDE_BIN_RESOLVED"

# --- Parse arguments ----------------------------------------------------------
NP=8
while [[ $# -gt 0 ]]; do
  case "$1" in
    --np) NP="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done
echo "MPI ranks: $NP"

# --- Shared input files (reuse from lbfgs_test) -------------------------------
LBFGS_DIR="$SCRIPT_DIR/../lbfgs_test"
if [[ ! -d "$LBFGS_DIR/other_decks" ]]; then
  echo "error: cannot find $LBFGS_DIR/other_decks/" >&2
  exit 1
fi

# Symlink other_decks if not present
if [[ ! -e "$SCRIPT_DIR/other_decks" ]]; then
  ln -s "$LBFGS_DIR/other_decks" "$SCRIPT_DIR/other_decks"
  echo "Symlinked other_decks -> $LBFGS_DIR/other_decks"
fi

# --- Scan definition ---------------------------------------------------------
# Format: "label w_l2 w_curl"
#
# Rationale:
#   - K/M entry ratio ~ 1e11 on this mesh (hz ~ 1.82 um)
#   - With equal weights, curl term dominates L2 by ~1e11
#   - "balanced" cases compensate with w_l2 = w_curl * 1e11
#
CASES=(
  # -- equal weights (curl dominates by ~1e11) --
  "l2_1e5_curl_1e5      1.0e5   1.0e5"

  # -- L2-only regularization --
  #"l2_1e5_curl_0        1.0e5   0.0"
  #"l2_1e11_curl_0       1.0e11  0.0"

  # -- curl-only regularization --
  #"l2_0_curl_1e5        0.0     1.0e5"
  #"l2_0_curl_1e11       0.0     1.0e11"

  # -- balanced: w_l2/w_curl ~ K/M ratio so both terms contribute --
  #"l2_1e16_curl_1e5     1.0e16  1.0e5"
  #"l2_1e5_curl_1e16     1.0e5   1.0e16"
  #"l2_1e11_curl_1       1.0e11  1.0"

  # -- intermediate ratios --
  #"l2_1e8_curl_1e5      1.0e8   1.0e5"
  #"l2_1e13_curl_1e5     1.0e13  1.0e5"

  # -- larger weights for both (stronger regularization overall) --
  #"l2_1e20_curl_1e9     1.0e20  1.0e9"
  #"l2_1e25_curl_1e14    1.0e25  1.0e14"
  #"l2_1e30_curl_1e19    1.0e30  1.0e19"
  #"l2_1e34_curl_1e23    1.0e34  1.0e23"
)

# --- Template generator -------------------------------------------------------
generate_input_yaml() {
  local w_l2="$1"
  local w_curl="$2"

  cat <<ENDYAML
%YAML 1.1
---
ANONYMOUS:
  debug level: 0
  verbosity: 0
  Mesh input file: other_decks/input_mesh.yaml
  Physics:
    modules: maxwell
    active controls: true
    Initial conditions:
      scalar data: true
      E: 0.0
      B: 0.0
    Dirichlet conditions:
      scalar data: false
      Ex:
        back: '0.0'
        front: '0.0'
      Ey:
        back: '0.0'
        front: '0.0'
      Ez:
        back: '0.0'
        front: '0.0'
    Mass weights:
      E: 1.0
      B: 1.0
    Norm weights:
      E: permittivity
      B: 1/permeability
  Discretization:
    order:
      E: 1
      B: 1
    quadrature: 2
    side quadrature: 2
  Parameters input file: other_decks/input_params.yaml
  Functions input file: other_decks/input_functions.yaml
  Solver:
    solver: 'transient'
    transient BDF order: 1
    transient Butcher tableau: DIRK-1,2
    workset size: 100
    final time: 2.0e-13
    number of steps: 400
    assembly partitioning: sequential
    use basis database: false
    use mass database: false
    database TOL: 1.0e-13
    linear TOL: 1.0e-14
    max linear iters: 200
    nonlinear TOL: 1.0e-07
    max nonlinear iters: 1
    use preconditioner: true
    preconditioner type: Ifpack2
    preconditioner variant: RELAXATION
    right preconditioner: true
    use direct solver: false
    reuse preconditioner: true
    reuse Jacobian: true
    preconditioner reuse type: full
    Belos implicit residual scaling: Norm of Initial Residual
    storage proportion: 1.0
    Belos solver: 'BiCGStab'
    enable autotune: true
    Preconditioner Settings:
      "relaxation: type": "Jacobi"
      "relaxation: damping factor": 0.5
      "relaxation: backward mode": false
      "relaxation: sweeps": 2
    dump jacobian: false
  Analysis input file: input_rol.yaml
  Postprocess:
    write solution: false
    write frequency: 1
    exodus write frequency: 1
    output file: opt-output/output
    compute objective: true
    compute response: false
    compute weighted norm: false
    compute sensitivities: false
    Objective functions:
      EM Energy:
        type: integrated control
        function: '0.5*permittivity*((E[x])^2+(E[y])^2+(E[z])^2) + 0.5/permeability*((B[x])^2+(B[y])^2+(B[z])^2)'
        weight: 1.0e35
        blocks: 'eblock-0_0_0 eblock-0_0_1 eblock-0_0_2 eblock-0_0_6 eblock-0_0_7 eblock-0_0_8'
      RegObj:
        type: integrated control
        function: '0.0'
        weight: 0.0
        blocks: 'eblock-0_0_3 eblock-0_0_5'
        Regularization functions:
          l2reg:
            type: integrated
            location: volume
            function: '0.5*(ctrl_current[x]*ctrl_current[x])+0.5*(ctrl_current[y]*ctrl_current[y])+0.5*(ctrl_current[z]*ctrl_current[z])'
            weight: ${w_l2}
          curlreg:
            type: integrated
            location: volume
            function: '0.5*(curl(ctrl_current)[x]*curl(ctrl_current)[x])+0.5*(curl(ctrl_current)[y]*curl(ctrl_current)[y])+0.5*(curl(ctrl_current)[z]*curl(ctrl_current)[z])'
            weight: ${w_curl}
    Extra cell fields:
      Jx: 'current x'
      Jy: 'current y'
      Jz: 'current z'
      ctrlJx: 'curl(ctrl_current)[x]'
      ctrlJy: 'curl(ctrl_current)[y]'
      ctrlJz: 'curl(ctrl_current)[z]'
      conductivity: sigma
      em_energy: '1.0e35 * (0.5*permittivity*(E[x]^2+E[y]^2+E[z]^2) + 0.5/permeability*(B[x]^2+B[y]^2+B[z]^2))'
...
ENDYAML
}

# --- Run loop -----------------------------------------------------------------
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

failed=0

for entry in "${CASES[@]}"; do
  read -r label w_l2 w_curl <<< "$entry"
  log="$LOG_DIR/mrhyde_${label}.log"

  echo ""
  echo "================================================================"
  echo "Run:     $label"
  echo "  w_l2:    $w_l2"
  echo "  w_curl:  $w_curl"
  echo "  Log:     $log"
  echo "================================================================"

  # Generate per-run input.yaml
  generate_input_yaml "$w_l2" "$w_curl" > input.yaml

  # Save a copy of the generated YAML for reproducibility
  cp input.yaml "yaml/input_${label}.yaml"

  mpiexec -n "$NP" "$MRHYDE_BIN" >"$log" 2>&1
  ec=$?

  echo "exit_code=$ec" >>"$log"

  if [[ $ec -ne 0 ]]; then
    echo "FAILED (exit $ec): $label" >&2
    failed=1
  else
    echo "OK: $label"
  fi
done

echo ""
echo "================================================================"
echo "All runs complete. Logs in: $LOG_DIR"
echo "================================================================"
exit "$failed"
