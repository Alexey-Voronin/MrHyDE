#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os, re
sys.path.append("../../scripts")
sys.path.append("../../../scripts/data_processing")
from mrhyde_test_support import *
from parse_log import solve_iterations, timer_calls, trust_region_rows

# ==============================================================================
# Parsing input

desc = ''' Solves the Riesz map's time slabs together. The transient Maxwell E-B
control case of param_riesz_curlcurl takes one trust-region step with the
iterative Riesz map three times: one slab per Krylov solve, two slabs per solve,
and the default, which solves all four slabs in one multi-column solve. The three
runs must take the same step, and each multi-column solve must need about as many
CG iterations as the slowest of its slabs solved alone. '''

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

DECKS = ['slab1', 'pairs', 'batch']
GROUPS = {'pairs': [2, 2], 'batch': [4]}   # slabs per solve in one four-slab apply
RIESZ_LIMIT = 500   # the Riesz CG's iteration limit; no other solve in these runs has it
CG_WINDOW = 3
lines = {}
for name in DECKS:
  log = 'mrhyde_%s.log' % name
  status += its.call('mpiexec -n 3 ../../mrhyde input_%s.yaml >& %s' % (name, log))
  its.clean_log(log)   # advisory IOSS filter; its sed call returns nonzero on BSD
  lines[name] = open(log, errors='replace').read().splitlines()

# Per group of every four-slab apply, the most iterations one of its slabs took alone.
def slowest(counts, sizes):
  out = []
  for a in range(0, len(counts), 4):
    for lo in range(len(sizes)):
      s0 = a + sum(sizes[:lo])
      out.append(max(counts[s0:s0 + sizes[lo]]))
  return out

steps = dict((name, trust_region_rows(lines[name])) for name in DECKS)
calls = dict((name, timer_calls(lines[name], 'MrHyDE::ParamRiesz::applyInverse()')) for name in DECKS)
cg = dict((name, solve_iterations(lines[name], RIESZ_LIMIT)) for name in DECKS)
if [] in steps.values() or None in calls.values() or [] in cg.values() or len(cg['slab1']) % 4:
  print('FAIL  could not read iteration 1, the applyInverse timer and four Riesz solves per apply from every log')
  status += 1
else:
  applies = len(cg['slab1']) // 4
  print('      %d Riesz applies; one slab per solve takes %d to %d CG iterations'
        % (applies, min(cg['slab1']), max(cg['slab1'])))
  for name in ['pairs', 'batch']:
    # One timed solve per group shows the slabs are batched.
    ok = calls[name] == applies * len(GROUPS[name])
    print('%s  %s applyInverse calls: %d, expected %d applies x %d groups'
          % ('pass' if ok else 'FAIL', name, calls[name], applies, len(GROUPS[name])))
    status += 0 if ok else 1

    want = slowest(cg['slab1'], GROUPS[name])
    ok = len(cg[name]) == len(want)
    off = max(abs(x - y) for x, y in zip(cg[name], want)) if ok else -1
    ok = ok and off <= CG_WINDOW
    print('%s  %s Riesz CG per solve: %d solves, at most %d from its slowest slab alone (window %d)'
          % ('pass' if ok else 'FAIL', name, len(cg[name]), off, CG_WINDOW))
    status += 0 if ok else 1

    s, r = steps['slab1'][0], steps[name][0]
    ok = s['iterCG'] == r['iterCG']
    print('%s  %s truncated-CG count: %d, one slab per solve %d' % ('pass' if ok else 'FAIL', name, r['iterCG'], s['iterCG']))
    status += 0 if ok else 1
    # ROL prints seven digits, so the window is 1e-6 relative.
    for label in ['value', 'gnorm', 'snorm']:
      x, y = s[label], r[label]
      rel = abs(x - y) / abs(x) if x != 0.0 else abs(y)
      ok = rel < 1.0e-6
      print('%s  %s iteration 1 %s: %.6e, one slab per solve %.6e, relative difference %.1e < 1e-6'
            % ('pass' if ok else 'FAIL', name, label, y, x, rel))
      status += 0 if ok else 1

if its.opts.clean and not status:
  status += its.call('rm -f mrhyde_slab1.log mrhyde_pairs.log mrhyde_batch.log')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
