#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os, re, random
sys.path.append("../../scripts")
from mrhyde_test_support import *

# ==============================================================================
# Parsing input

desc = ''' Control on the Riesz map's rows only: gradient and Hessian vanish off them,
a written control reloads exactly, and the guards stop. '''

its = mrhyde_test_support(desc)

its.opts.verbose = True

root = 'mrhyde'

#TESTING active
#TESTING -n 2
#TESTING -k maxwell,optimization,riesz,hcurl,parameter

# ==============================================================================
status = 0

if its.opts.preprocess:
  status += its.call('echo "  No preprocessing, yet."')

# Random control on every row and slab.
NSP, NSLABS = 980, 4
rng = random.Random(11)
for s in range(NSLABS):
  with open('v.field.%d.mm' % s, 'w') as f:
    f.write('%%%%MatrixMarket matrix array real general\n%d 1\n' % NSP)
    for _ in range(NSP):
      f.write('%.17e\n' % (1.0e-3 * rng.gauss(0.0, 1.0)))

DECKS = ['gradient', 'hessian', 'write', 'reload', 'guard', 'forward', 'rolstoch']
STOPS = ['guard', 'rolstoch']   # expected to abort
logs = {}
for name in DECKS:
  log = 'mrhyde_%s.log' % name
  cmd = 'mpiexec -n 2 ../../mrhyde input_%s.yaml >& %s' % (name, log)
  status += its.call(cmd + (' || true' if name in STOPS else ''))
  its.clean_log(log)
  logs[name] = open(log, errors='replace').read()

def check(ok, what):
  print('%s  %s' % ('pass' if ok else 'FAIL', what))
  return 0 if ok else 1

def numbers(path):
  return [l for l in open(path).read().splitlines() if l and not l.startswith('%')]

# Control rows: nonzero metric diagonal (1-based ids).
ctrl = set()
for l in numbers('riesz.mm')[1:]:
  i, j, v = l.split()
  if i == j and float(v) != 0.0:
    ctrl.add(int(i) - 1)
m = re.search(r'\[RieszMap\] P = .* dim (\d+), control rows (\d+)', logs['gradient'])
status += check(m is not None and int(m.group(2)) < int(m.group(1)) and int(m.group(2)) == len(ctrl),
                'control rows: fewer than parameter rows, one per metric diagonal entry')

# Dense files list rows in hess.map order.
gids = [int(v) for v in numbers('hess_exact.map')[1::2]]
off, on = 0.0, 0.0
for t in range(NSLABS):
  vals = [float(v) for v in numbers('grad.field.%d.mm' % t)[1:]]
  for g, v in zip(gids, vals):
    if g in ctrl: on = max(on, abs(v))
    else:         off = max(off, abs(v))
status += check(len(gids) > len(ctrl) and on > 0.0 and off == 0.0,
                'gradient zero off the control rows (%.1e)' % off)

# Column j = slab*n + parameter id.
n = len(gids)
cols = [int(v) for v in open('hess_exact.cols').read().split()]
row = dict((g, k) for k, g in enumerate(gids))
blocks = []
for t in range(NSLABS):
  vals = [float(v) for v in numbers('hess_exact_slab%d.mtx' % t)[1:]]
  blocks.append([vals[c*n:(c+1)*n] for c in range(len(cols))])
def H(i, c):   # global row i, dump column c
  return blocks[i // n][c][row[i % n]]
top = max(abs(v) for b in blocks for col in b for v in col) if cols else 0.0
asym = max([abs(H(cols[a], b) - H(cols[b], a)) for a in range(len(cols)) for b in range(len(cols))] or [1.0])
offrow = max([abs(col[k]) for b in blocks for col in b for k, g in enumerate(gids) if g not in ctrl] or [0.0])
status += check(len(cols) >= 2 and top > 0.0 and asym <= 1.0e-10 * top,
                'Hessian block symmetric (%.1e of max)' % (asym / top if top else 0.0))
status += check(offrow == 0.0, 'Hessian zero off the control rows (%.1e)' % offrow)

# Files keep every bit, so the round trip matches exactly.
same = lambda a, b: open(a, 'rb').read() == open(b, 'rb').read()
status += check(same('obj_write.dat', 'obj_reload.dat')
                and all(same('gw.field.%d.mm' % t, 'gr.field.%d.mm' % t) for t in range(NSLABS))
                and all(same('pw.field.%d.mm' % t, 'pr.field.%d.mm' % t) for t in range(NSLABS)),
                'reload reproduces objective, gradient and control files')
dev = 0.0
for t in range(NSLABS):
  read = [float(v) for v in numbers('v.field.%d.mm' % t)[1:]]
  wrote = [float(v) for v in numbers('pw.field.%d.mm' % t)[1:]]
  dev = max([dev] + [abs(w - (r if g in ctrl else 0.0)) for g, r, w in zip(gids, read, wrote)])
status += check(dev == 0.0, 'written control = read one on control rows, 0 elsewhere (%.1e)' % dev)

G1 = "the control acts outside the Riesz map's blocks"
status += check(logs['guard'].count(G1) == 2, 'guard stops on both ranks')
status += check('[RieszMap] not built' in logs['forward'] and 'Riesz map: analysis type' not in logs['forward'],
                'forward builds no metric')
status += check(open('obj_forward.dat').read() == open('obj_gradient.dat').read(),
                'forward objective = forward+adjoint objective')
status += check("analysis type 'ROLStoch' applies the metric" in logs['rolstoch'], 'ROLStoch stops')

if its.opts.clean and not status:
  status += its.call('rm -f mrhyde_*.log riesz.mm *.field.*.mm hess_exact* obj_*.dat')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
