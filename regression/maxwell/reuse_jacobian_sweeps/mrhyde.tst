#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os, re
sys.path.append("../../scripts")
from mrhyde_test_support import *

# ==============================================================================
# Parsing input

desc = ''' Direct solves with 'reuse Jacobian across sweeps' against the same deck
without it. The LQ Maxwell checks run forward, adjoint and both incremental sweeps,
so the kept forward and adjoint factorizations are both reused; the exact-path
numbers must not move. The mass run adds 'print mass', whose setup leaves a mass
matrix in the state context that must not be kept as the Jacobian. The amesos
run passes 'Amesos2 Settings' through to KLU2 and must not move a number either. '''

its = mrhyde_test_support(desc)

its.opts.verbose = True

root = 'mrhyde'

#TESTING active
#TESTING -n 3
#TESTING -k maxwell,optimization,hessvec,direct

# ==============================================================================
status = 0

if its.opts.preprocess:
  status += its.call('echo "  No preprocessing, yet."')

logs = {}
for name in ('off', 'on', 'mass', 'amesos'):
  log = 'mrhyde_%s.log' % name
  status += its.call('mpiexec -n 3 ../../mrhyde input_%s.yaml >& %s' % (name, log))
  its.clean_log(log)   # advisory IOSS filter; its sed call returns nonzero on BSD
  logs[name] = open(log, errors='replace').read()
status += its.call('rm -f mass.mm lumped_mass.mm')

NUM = r'([-+]?\d+\.\d+[eE][-+]?\d+)'
ROW = re.compile(r'^\s+' + r'\s+'.join([NUM] * 4) + r'\s*$', re.M)

# Second column of a check table: grad'*dir and norm(Hess*vec) are exact-path values.
def _table_column(text, header, nxt):
  i = text.find(header)
  j = text.find(nxt, i + 1)
  if i < 0 or j < 0:
    return []
  return [float(m.group(2)) for m in ROW.finditer(text[i:j])]

def _exact_numbers(text):
  out = _table_column(text, 'Gradient check', 'Hessian-vector check')
  out += _table_column(text, 'Hessian-vector check', 'Hessian symmetry check')
  m = re.search(r'Hessian symmetry check.*?abs error\s*\n\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)',
                text, re.DOTALL)
  if m:
    out += [float(m.group(1)), float(m.group(2))]
  m = re.search(r'<v1, H v1>\s*=\s*([-+0-9.eE]+).*?<v2, H v2>\s*=\s*([-+0-9.eE]+)', text)
  if m:
    out += [float(m.group(1)), float(m.group(2))]
  m = re.search(r'^\s*0\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+---', text, re.M)
  if m:
    out += [float(m.group(1)), float(m.group(2))]
  return out

# Parse "relative =" scalar for a tag.
def _second_scalar(tag):
  def _e(text):
    m = re.search(re.escape(tag) + r'.*?relative\s*=\s*([-+0-9.eE]+)', text)
    if m is None:
      return None
    try:
      return float(m.group(1))
    except ValueError:
      return None
  return _e

checks = [
  ('[HV-ZERO] || H*0 || == 0',
   pred_scalar_lt('[HV-ZERO] || H*0 || =', 1.0e-30)),
  ('[HV-BILINEARITY] relative < 1e-10',
   pred_scalar_lt('[HV-BILINEARITY]', 1.0e-10, extractor=_second_scalar('[HV-BILINEARITY]'))),
  ('[SECANT-IDENTITY] relative < 1e-12',
   pred_scalar_lt('[SECANT-IDENTITY]', 1.0e-12, extractor=_second_scalar('[SECANT-IDENTITY]'))),
  ('ROL terminates at Iteration Limit',
   pred_contains('Iteration Limit Exceeded')),
]
status += check_predicates('mrhyde_on.log', checks, verbose=True)

# Every run factors the same matrices, so a reused factorization changes nothing.
a = _exact_numbers(logs['off'])
for name in ('on', 'mass', 'amesos'):
  b = _exact_numbers(logs[name])
  if len(a) != 32 or len(b) != len(a):
    print('FAIL  exact-path numbers: read %d and %d of 32' % (len(a), len(b)))
    status += 1
    continue
  worst = max(abs(x - y) / max(abs(x), 1.0e-300) for x, y in zip(a, b))
  ok = worst <= 1.0e-12
  print('%s  32 exact-path numbers, %s against off: max rel diff %.3e <= 1e-12'
        % ('pass' if ok else 'FAIL', name, worst))
  if not ok:
    status += 1

# 'Amesos2 Settings' reaches KLU2, which prints a header on every solve at DebugLevel 1.
ok = '== Amesos2_KLU2::solve_impl ==' in logs['amesos']
print('%s  Amesos2 Settings reached KLU2 (DebugLevel 1 solve header in the log)' % ('pass' if ok else 'FAIL'))
status += 0 if ok else 1

# Same pair, iterative: Jacobi draws no random start vector, so kept and rebuilt preconditioners agree.
sys.path.append("../../../scripts/data_processing")
from parse_log import timer_calls

for name in ('iter_off', 'iter_on'):
  log = 'mrhyde_%s.log' % name
  status += its.call('mpiexec -n 3 ../../mrhyde input_%s.yaml >& %s' % (name, log))
  its.clean_log(log)   # advisory IOSS filter; its sed call returns nonzero on BSD
  logs[name] = open(log, errors='replace').read()

a = _exact_numbers(logs['iter_off'])
b = _exact_numbers(logs['iter_on'])
if len(a) != 32 or len(b) != len(a):
  print('FAIL  iterative exact-path numbers: read %d and %d of 32' % (len(a), len(b)))
  status += 1
else:
  worst = max(abs(x - y) / max(abs(x), 1.0e-300) for x, y in zip(a, b))
  ok = worst <= 1.0e-12
  print('%s  32 exact-path numbers, iter_on against iter_off: max rel diff %.3e <= 1e-12'
        % ('pass' if ok else 'FAIL', worst))
  status += 0 if ok else 1

LABELS = ('Ifpack2::Relaxation::compute', 'Ifpack2::Relaxation::apply',
          'MrHyDE::SolverManager::forward()', 'MrHyDE::SolverManager::adjoint()')
calls = {}
for name in ('iter_off', 'iter_on'):
  lines = logs[name].splitlines()
  calls[name] = [timer_calls(lines, label) for label in LABELS]
if None in calls['iter_off'] + calls['iter_on']:
  print('FAIL  could not read %s from both iterative logs' % ', '.join(LABELS))
  status += 1
else:
  build_off, apply_off, fwd_off, adj_off = calls['iter_off']
  build_on, apply_on, fwd_on, adj_on = calls['iter_on']
  ok = (fwd_on, adj_on) == (fwd_off, adj_off)
  print('%s  forward/adjoint sweeps: %d/%d with the flag, %d/%d without'
        % ('pass' if ok else 'FAIL', fwd_on, adj_on, fwd_off, adj_off))
  status += 0 if ok else 1
  # One Jacobi setup per kept slot (J, J^T) with the flag; one per sweep without it.
  ok = 1 <= build_on <= 2 < build_off
  print('%s  Jacobi setups: %d with the flag (at most one per kept slot), %d without'
        % ('pass' if ok else 'FAIL', build_on, build_off))
  status += 0 if ok else 1
  # Teuchos prints call counts above 9999 to four significant digits.
  ok = abs(apply_on - apply_off) <= 1.0e-3 * apply_off
  print('%s  Jacobi applies: %d with the flag, %d without'
        % ('pass' if ok else 'FAIL', apply_on, apply_off))
  status += 0 if ok else 1

# A BDF order switch changes the operator, so the kept Jacobian must be refused, not reused.
its.call('mpiexec -n 3 ../../mrhyde input_iter_bdf2.yaml >& mrhyde_iter_bdf2.log', ignore_status=True)
status += check_predicates('mrhyde_iter_bdf2.log',
                           [('stale kept Jacobian refused', pred_contains('reuse Jacobian across sweeps:'))],
                           verbose=True)

if its.opts.clean and not status:
  status += its.call('rm -f mrhyde_iter_off.log mrhyde_iter_on.log mrhyde_iter_bdf2.log')

if its.opts.clean and not status:
  status += its.call('rm -f mrhyde_off.log mrhyde_on.log mrhyde_mass.log mrhyde_amesos.log')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
