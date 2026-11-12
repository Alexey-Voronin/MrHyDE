#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os, re, random
sys.path.append("../../scripts")
sys.path.append("../../../scripts/data_processing")
from mrhyde_test_support import *
from parse_log import timer_calls

# ==============================================================================
# Parsing input

desc = ''' One block-LU pass per E-B solve ('triangle: full', inner CG on both splits,
flexible Block GMRES) must take the direct solver's trust-region step in one outer iteration. '''

its = mrhyde_test_support(desc)

its.opts.verbose = True

root = 'mrhyde'

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,optimization,blocktri,exact-curl,adjoint

# ==============================================================================
status = 0

if its.opts.preprocess:
  status += its.call('echo "  No preprocessing, yet."')

# Random control per time slab: a constant one lies in the misfit's kernel and skips the adjoint.
NSP, NSLABS, SEED = 224, 8, 11
rng = random.Random(SEED)
for s in range(NSLABS):
  with open('v.field.%d.mm' % s, 'w') as f:
    f.write('%%%%MatrixMarket matrix array real general\n%d 1\n' % NSP)
    for _ in range(NSP):
      f.write('%.17e\n' % (1.0e-3 * rng.gauss(0.0, 1.0)))

DECKS = ['direct', 'lu']
logs = {}
for name in DECKS:
  log = 'mrhyde_%s.log' % name
  status += its.call('mpiexec -n 4 ../../mrhyde input_%s.yaml >& %s' % (name, log))
  its.clean_log(log)   # advisory IOSS filter; its sed call returns nonzero on BSD
  logs[name] = open(log, errors='replace').read()

# CG count and flag of trust-region step 1 (columns iterCG, flagCG).
def _step1(text):
  for line in text.splitlines():
    g = line.split()
    if len(g) == 10 and g[0] == '1' and g[8].isdigit():
      return int(g[8]), int(g[9])
  return None, None

# ROL's gradient norm at iteration 0.
def _iter0_gnorm(text):
  m = re.search(r'^\s*0\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+---', text, re.M)
  return None if m is None else float(m.group(2))

common = [
  ('[GRAD-CHECK] best rel_err < 1e-3',
   pred_scalar_lt('[GRAD-CHECK] best rel_err =', 1.0e-3)),
]
for name in DECKS:
  print('--- %s ---' % name)
  status += check_predicates('mrhyde_%s.log' % name, common, verbose=True)

cg, flag = {}, {}
for name in DECKS:
  cg[name], flag[name] = _step1(logs[name])
  ok = flag[name] == 0
  print('%s  %s CG converged (flag 0): CG %s flag %s' % ('pass' if ok else 'FAIL', name, cg[name], flag[name]))
  if not ok:
    status += 1

if cg['direct'] is not None and cg['lu'] is not None:
  ok = abs(cg['lu'] - cg['direct']) <= 3
  print('%s  |CG block LU - CG direct| <= 3: %d vs %d' % ('pass' if ok else 'FAIL', cg['lu'], cg['direct']))
  if not ok:
    status += 1

g = dict((name, _iter0_gnorm(logs[name])) for name in DECKS)
if None in g.values() or g['direct'] == 0.0:
  print('FAIL  could not read the iteration-0 gradient norm from both logs')
  status += 1
else:
  rel = abs(g['lu'] - g['direct']) / g['direct']
  ok = rel < 1.0e-5
  print('%s  gradient norm matches direct to 1e-5: %.7e vs %.7e (rel %.2e)'
        % ('pass' if ok else 'FAIL', g['lu'], g['direct'], rel))
  if not ok:
    status += 1

# Inner solves are CG and the Riesz map is factored, so ICGS calls = outer solves + outer iterations.
lines = logs['lu'].splitlines()
ortho = timer_calls(lines, 'Belos: ICGS[2]: Orthogonalization')
solves = timer_calls(lines, 'MrHyDE::LinearAlgebraInterface::linearSolver*()')
if not ortho or not solves:
  print('FAIL  could not read the outer GMRES and solve counts from mrhyde_lu.log')
  status += 1
else:
  outer = float(ortho - solves) / solves
  ok = outer <= 2.0
  print('%s  outer GMRES iterations per solve <= 2: %.2f over %d solves' % ('pass' if ok else 'FAIL', outer, solves))
  if not ok:
    status += 1

if its.opts.clean and not status:
  status += its.call('rm -f mrhyde_direct.log mrhyde_lu.log v.field.*.mm objective.txt')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
