#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os, re
sys.path.append("../../scripts")
sys.path.append("../../../scripts/data_processing")
from mrhyde_test_support import *
from parse_log import timer_calls

desc = ''' 'reuse Jacobian across sweeps' against the same deck without it, with direct and
iterative solves: the LQ Maxwell checks (forward, adjoint and both incremental sweeps)
must not move, the kept operators are built once, and a BDF order switch must refuse
the kept Jacobian. '''

its = mrhyde_test_support(desc)
its.opts.verbose = True

#TESTING active
#TESTING -n 3
#TESTING -k maxwell,optimization,hessvec,direct

# ==============================================================================
status = 0

if its.opts.preprocess:
  status += its.call('echo "  No preprocessing, yet."')

def check(ok, msg):
  global status
  print('%s  %s' % ('pass' if ok else 'FAIL', msg))
  status += 0 if ok else 1

# 'mass' adds 'print mass', whose setup leaves a mass matrix in the state context that must not be kept as the
# Jacobian; 'amesos' passes 'Amesos2 Settings' through to KLU2.
logs = {}
for name in ('off', 'on', 'mass', 'amesos', 'iter_off', 'iter_on'):
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
  for pattern, flags in ((r'Hessian symmetry check.*?abs error\s*\n\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)', re.DOTALL),
                         (r'<v1, H v1>\s*=\s*([-+0-9.eE]+).*?<v2, H v2>\s*=\s*([-+0-9.eE]+)', 0),
                         (r'^\s*0\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+---', re.M)):
    m = re.search(pattern, text, flags)
    if m:
      out += [float(m.group(1)), float(m.group(2))]
  return out

# Parse "relative =" scalar for a tag.
def _second_scalar(tag):
  def _e(text):
    m = re.search(re.escape(tag) + r'.*?relative\s*=\s*([-+0-9.eE]+)', text)
    try:
      return float(m.group(1)) if m else None
    except ValueError:
      return None
  return _e

status += check_predicates('mrhyde_on.log', [
  ('[HV-ZERO] || H*0 || == 0', pred_scalar_lt('[HV-ZERO] || H*0 || =', 1.0e-30)),
  ('[HV-BILINEARITY] relative < 1e-10',
   pred_scalar_lt('[HV-BILINEARITY]', 1.0e-10, extractor=_second_scalar('[HV-BILINEARITY]'))),
  ('[SECANT-IDENTITY] relative < 1e-12',
   pred_scalar_lt('[SECANT-IDENTITY]', 1.0e-12, extractor=_second_scalar('[SECANT-IDENTITY]'))),
  ('ROL terminates at Iteration Limit', pred_contains('Iteration Limit Exceeded')),
], verbose=True)

# Every run factors or preconditions the same matrices, so a kept operator changes nothing. Jacobi draws no random
# start vector, so kept and rebuilt iterative preconditioners agree too.
for ref, name in (('off', 'on'), ('off', 'mass'), ('off', 'amesos'), ('iter_off', 'iter_on')):
  a, b = _exact_numbers(logs[ref]), _exact_numbers(logs[name])
  if len(a) != 32 or len(b) != 32:
    check(False, 'exact-path numbers of %s and %s: read %d and %d of 32' % (ref, name, len(a), len(b)))
    continue
  worst = max(abs(x - y) / max(abs(x), 1.0e-300) for x, y in zip(a, b))
  check(worst <= 1.0e-12, '32 exact-path numbers, %s against %s: max rel diff %.3e <= 1e-12' % (name, ref, worst))

# KLU2 prints a header on every solve at DebugLevel 1.
check('== Amesos2_KLU2::solve_impl ==' in logs['amesos'], 'Amesos2 Settings reached KLU2 (DebugLevel 1 solve header)')

LABELS = ('Ifpack2::Relaxation::compute', 'Ifpack2::Relaxation::apply',
          'MrHyDE::SolverManager::forward()', 'MrHyDE::SolverManager::adjoint()',
          'MrHyDE::LinearAlgebraInterface::fillComplete*()')
calls = dict((n, [timer_calls(logs[n].splitlines(), label) for label in LABELS]) for n in ('iter_off', 'iter_on'))
if None in calls['iter_off'] + calls['iter_on']:
  check(False, 'could not read %s from both iterative logs' % ', '.join(LABELS))
else:
  build_off, apply_off, fwd_off, adj_off, fill_off = calls['iter_off']
  build_on, apply_on, fwd_on, adj_on, fill_on = calls['iter_on']
  check((fwd_on, adj_on) == (fwd_off, adj_off),
        'forward/adjoint sweeps: %d/%d with the flag, %d/%d without' % (fwd_on, adj_on, fwd_off, adj_off))
  # One Jacobi setup per kept slot (J, J^T) with the flag; one per sweep without it.
  check(1 <= build_on <= 2 < build_off,
        'Jacobi setups: %d with the flag (at most one per kept slot), %d without' % (build_on, build_off))
  # A kept previous-step Jacobian is assembled once per run; rebuilt, it costs 3 fillComplete calls per adjoint sweep.
  check(fill_on < 3 * adj_on, 'fillComplete calls with the flag: %d, below 3 per adjoint sweep (%d)' % (fill_on, 3 * adj_on))
  # Teuchos prints call counts above 9999 to four significant digits.
  check(abs(apply_on - apply_off) <= 1.0e-3 * apply_off,
        'Jacobi applies: %d with the flag, %d without' % (apply_on, apply_off))

# A BDF order switch changes the operator, so the kept Jacobian must be refused, not reused.
its.call('mpiexec -n 3 ../../mrhyde input_iter_bdf2.yaml >& mrhyde_iter_bdf2.log', ignore_status=True)
status += check_predicates('mrhyde_iter_bdf2.log',
                           [('stale kept Jacobian refused', pred_contains('reuse Jacobian across sweeps:'))],
                           verbose=True)

if its.opts.clean and not status:
  status += its.call('rm -f mrhyde_off.log mrhyde_on.log mrhyde_mass.log mrhyde_amesos.log '
                     'mrhyde_iter_off.log mrhyde_iter_on.log mrhyde_iter_bdf2.log')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
