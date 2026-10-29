#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os, re
sys.path.append("../../scripts")
sys.path.append("../../../scripts/data_processing")
from mrhyde_test_support import *
from parse_log import solve_iterations, trust_region_rows

# ==============================================================================
# Parsing input

desc = ''' Moves the Riesz map's control rows to other ranks. The transient Maxwell
E-B control case of param_riesz_curlcurl takes one trust-region step with the
iterative Riesz map three times: on the mesh layout (ranks: -1), on the default
layout, which puts this small control block on one rank, and on all three ranks
(ranks: 3). The three runs must take the same step, and the moved rows must not
cost the Riesz CG more iterations than the mesh layout. '''

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

DECKS = ['mesh', 'auto', 'three']
RIESZ_LIMIT = 500   # the Riesz CG's iteration limit; no other solve in these runs has it
CG_WINDOW = 3
logs = {}
for name in DECKS:
  log = 'mrhyde_%s.log' % name
  status += its.call('mpiexec -n 3 ../../mrhyde input_%s.yaml >& %s' % (name, log))
  its.clean_log(log)   # advisory IOSS filter; its sed call returns nonzero on BSD
  logs[name] = open(log, errors='replace').read()

# Ranks holding control rows, from the [RieszMap] layout line (verbosity 5).
holders = dict((name, extract_scalar(logs[name], '[RieszMap] control rows on')) for name in DECKS)
steps = dict((name, trust_region_rows(logs[name].splitlines())) for name in DECKS)
cg = dict((name, solve_iterations(logs[name].splitlines(), RIESZ_LIMIT)) for name in DECKS)
if None in holders.values() or [] in steps.values() or [] in cg.values():
  print('FAIL  could not read the [RieszMap] layout line, iteration 1 and the Riesz CG counts from every log')
  status += 1
else:
  print('      mesh layout: control rows on %d of 3 ranks, Riesz CG %d to %d per solve'
        % (holders['mesh'], min(cg['mesh']), max(cg['mesh'])))
  # The default gives this small control block one rank; ranks: 3 spreads it over all three.
  for name, want in [('auto', 1), ('three', 3)]:
    ok = int(holders[name]) == want
    print('%s  %s: control rows on %d of 3 ranks, expected %d'
          % ('pass' if ok else 'FAIL', name, holders[name], want))
    status += 0 if ok else 1

  for name in ['auto', 'three']:
    # A coordinate or nullspace entry moved to the wrong row weakens RefMaxwell and shows up here.
    ok = len(cg[name]) == len(cg['mesh'])
    over = max(x - y for x, y in zip(cg[name], cg['mesh'])) if ok else -1
    ok = ok and over <= CG_WINDOW
    print('%s  %s Riesz CG: %d solves, %d to %d per solve, at most %d over the mesh layout (window %d)'
          % ('pass' if ok else 'FAIL', name, len(cg[name]), min(cg[name]), max(cg[name]), over, CG_WINDOW))
    status += 0 if ok else 1

    s, r = steps['mesh'][0], steps[name][0]
    ok = s['iterCG'] == r['iterCG']
    print('%s  %s truncated-CG count: %d, mesh layout %d' % ('pass' if ok else 'FAIL', name, r['iterCG'], s['iterCG']))
    status += 0 if ok else 1
    # ROL prints seven digits, so the window is 1e-6 relative.
    for label in ['value', 'gnorm', 'snorm']:
      x, y = s[label], r[label]
      rel = abs(x - y) / abs(x) if x != 0.0 else abs(y)
      ok = rel < 1.0e-6
      print('%s  %s iteration 1 %s: %.6e, mesh layout %.6e, relative difference %.1e < 1e-6'
            % ('pass' if ok else 'FAIL', name, label, y, x, rel))
      status += 0 if ok else 1

if its.opts.clean and not status:
  status += its.call('rm -f mrhyde_mesh.log mrhyde_auto.log mrhyde_three.log')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
