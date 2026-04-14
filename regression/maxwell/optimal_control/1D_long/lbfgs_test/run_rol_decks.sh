#!/usr/bin/env bash
#
# Run mrhyde once per rol_decks/input_rol2_*.yaml (skips rol_decks/all/).
# Copies each deck to input_rol.yaml; uses 8 MPI ranks.
#


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR" || exit 1

MRHYDE_ROOT="$(cd "$SCRIPT_DIR/../../../../../../" && pwd -P)"
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



shopt -s nullglob
decks=(rol_decks/input_rol2_*.yaml)
shopt -u nullglob

if [[ ${#decks[@]} -eq 0 ]]; then
  echo "error: no rol_decks/input_rol2_*.yaml found" >&2
  exit 1
fi

failed=0

for deck in "${decks[@]}"; do
  base=$(basename "$deck" .yaml)
  suffix=${base#input_rol2_}
  log="mrhyde_${suffix}.log"

  echo ""
  echo "Running:  $deck"
  echo "Log file: $log"

  cp "$deck" input_rol.yaml
  ROL_LBFGS_DIAG=1 ROL_LBFGS_RESET_AT=-1 ROL_LBFGS_RESET_PERIOD=-1 mpiexec -n 8 "$MRHYDE_BIN" >"$log" 2>&1
  ec=$?

  echo "exit_code=$ec" >>"$log"

  if [[ $ec -ne 0 ]]; then
    echo "FAILED (exit $ec): $deck" >&2
    failed=1
  else
    echo "OK: $deck"
  fi
done

echo ""
exit "$failed"
