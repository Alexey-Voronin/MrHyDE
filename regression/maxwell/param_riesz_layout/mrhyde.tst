#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os, re
sys.path.append("../../scripts")
sys.path.append("../../../scripts/data_processing")
from mrhyde_test_support import *
from parse_log import solve_iterations, trust_region_rows

# ==============================================================================
# Parsing input

desc = ''' The Riesz map's control rows on the mesh layout (ranks: -1), spread over every
rank (the default) and gathered on one (ranks: 1) must give the same trust-region
step, and moving them must not raise the Riesz CG count. '''

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

DECKS = ['mesh', 'auto', 'one']
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
  s = steps['mesh'][0]
  for name, want in [('auto', 3), ('one', 1)]:
    r = steps[name][0]
    # A coordinate or nullspace entry moved to the wrong row weakens RefMaxwell and raises the CG count.
    over = max(x - y for x, y in zip(cg[name], cg['mesh'])) if len(cg[name]) == len(cg['mesh']) else None
    # ROL prints seven digits, so the window is 1e-6 relative.
    rel = max(abs(s[k] - r[k]) / abs(s[k]) if s[k] != 0.0 else abs(r[k]) for k in ('value', 'gnorm', 'snorm'))
    for ok, what in [(int(holders[name]) == want, 'control rows on %d of 3 ranks, expected %d' % (holders[name], want)),
                     (over is not None and over <= CG_WINDOW,
                      'Riesz CG at most %s over the mesh layout (window %d)' % (over, CG_WINDOW)),
                     (r['iterCG'] == s['iterCG'], 'truncated CG %d iterations, mesh layout %d' % (r['iterCG'], s['iterCG'])),
                     (rel < 1.0e-6, 'iteration 1 value, gnorm and snorm within %.1e of the mesh layout (< 1e-6)' % rel)]:
      print('%s  %s: %s' % ('pass' if ok else 'FAIL', name, what))
      status += 0 if ok else 1

if its.opts.clean and not status:
  status += its.call('rm -f mrhyde_mesh.log mrhyde_auto.log mrhyde_one.log')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
