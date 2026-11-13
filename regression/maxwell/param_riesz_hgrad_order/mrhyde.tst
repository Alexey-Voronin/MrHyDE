#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os, re
sys.path.append("../../scripts")
sys.path.append("../../../scripts/data_processing")
from mrhyde_test_support import *
from parse_log import solve_iterations, trust_region_rows

# ==============================================================================
# Parsing input

desc = ''' The iterative Riesz map's RefMaxwell takes its H(grad) space at the
control's order unless 'hgrad order' sets it. The transient Maxwell E-B control
case of param_riesz_batch, with the control raised to order 2 and the exact-curl
block solver for the state, takes one trust-region step twice: with the default,
H(grad) at order 2, and with 'hgrad order: 1'. Every linear solve must converge,
both must take the same step, and the default must need fewer Riesz CG
iterations per solve. '''

its = mrhyde_test_support(desc)

its.opts.verbose = True

root = 'mrhyde'

#TESTING active
#TESTING -n 3
#TESTING -k maxwell,optimization,riesz,hcurl,parameter

# ==============================================================================
status = 0

if its.opts.preprocess:
  status += its.call('echo "  No preprocessing, yet."')

DECKS = ['control', 'one']
RIESZ_LIMIT = 500   # the Riesz CG's iteration limit; no other solve in these runs has it
lines = {}
for name in DECKS:
  log = 'mrhyde_%s.log' % name
  status += its.call('mpiexec -n 3 ../../mrhyde input_%s.yaml >& %s' % (name, log))
  its.clean_log(log)   # advisory IOSS filter; its sed call returns nonzero on BSD
  lines[name] = open(log, errors='replace').read().splitlines()

# A state solve short of its tolerance would make the step comparison meaningless.
for name in DECKS:
  missed = sum(1 for l in lines[name] if 'did not converge' in l)
  ok = missed == 0
  print('%s  %s: %d linear solves missed their tolerance' % ('pass' if ok else 'FAIL', name, missed))
  status += 0 if ok else 1

steps = dict((name, trust_region_rows(lines[name])) for name in DECKS)
cg = dict((name, solve_iterations(lines[name], RIESZ_LIMIT)) for name in DECKS)
if [] in steps.values() or [] in cg.values():
  print('FAIL  could not read iteration 1 and the Riesz solves from every log')
  status += 1
else:
  ok = len(cg['control']) == len(cg['one'])
  print('%s  Riesz solves: %d with the default, %d with hgrad order 1'
        % ('pass' if ok else 'FAIL', len(cg['control']), len(cg['one'])))
  status += 0 if ok else 1

  # Strictly fewer also shows the key reached the solver.
  mc = float(sum(cg['control'])) / len(cg['control'])
  mo = float(sum(cg['one'])) / len(cg['one'])
  ok = mc < mo
  print('%s  Riesz CG per solve: %.2f with the default, %.2f with hgrad order 1' % ('pass' if ok else 'FAIL', mc, mo))
  status += 0 if ok else 1

  s, r = steps['one'][0], steps['control'][0]
  ok = s['iterCG'] == r['iterCG']
  print('%s  truncated-CG count: %d, hgrad order 1 %d' % ('pass' if ok else 'FAIL', r['iterCG'], s['iterCG']))
  status += 0 if ok else 1
  # ROL prints seven digits, so the window is 1e-6 relative.
  for label in ['value', 'gnorm', 'snorm']:
    x, y = s[label], r[label]
    rel = abs(x - y) / abs(x) if x != 0.0 else abs(y)
    ok = rel < 1.0e-6
    print('%s  iteration 1 %s: %.6e, hgrad order 1 %.6e, relative difference %.1e < 1e-6'
          % ('pass' if ok else 'FAIL', label, y, x, rel))
    status += 0 if ok else 1

if its.opts.clean and not status:
  status += its.call('rm -f mrhyde_control.log mrhyde_one.log')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
