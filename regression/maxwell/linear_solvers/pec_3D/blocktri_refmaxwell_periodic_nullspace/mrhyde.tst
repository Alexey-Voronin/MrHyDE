#!/usr/bin/env python3

import re
import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check, Results

its = mrhyde_test_support('''RefMaxwell on a cavity periodic in z, refined in h at a large fixed dt.
D0 times the nodal coordinates wraps across the periodic face and misses the constant z field, so
MrHyDE supplies the edge nullspace; with MueLu's own one the iteration counts climb out of band.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k regression,maxwell,HCURL,HDIV,blocktriangular,schur,schur_hcurl,refmaxwell,crossrun,periodic

MESHES = ["N8x8x4", "N16x8x4", "N24x12x6"]
NXNYNZ = {"N8x8x4": (8, 8, 4), "N16x8x4": (16, 8, 4), "N24x12x6": (24, 12, 6)}
# (mean, max) iterations of the two solves on each mesh.
CASES = {"N8x8x4": (12.0, 12), "N16x8x4": (15.5, 16), "N24x12x6": (17.0, 17)}

res = Results()
status = enable_trilinos_debug(tpetra=False)
for mesh in MESHES:
    log = "mrhyde_%s.log" % mesh
    status += its.call("mpiexec -n 4 ../../../../mrhyde input_%s.yaml >& %s" % (mesh, log))
    text = open(log).read()
    built = tuple(int(re.search(r"^ %s = (\d+)$" % n, text, re.M).group(1))
                  for n in ("NX", "NY", "NZ"))
    res.add(built == NXNYNZ[mesh], "%s mesh built" % mesh,
            "NX/NY/NZ %s, expected %s" % (built, NXNYNZ[mesh]))
    supplied = "building edge nullspace" not in text
    res.add(supplied, "%s nullspace supplied" % mesh,
            "from element-local coordinates" if supplied
            else "MueLu built D0 times coords itself, which is wrong across the periodic face")
    mean, imax = CASES[mesh]
    check(solves=2, mean=mean, imax=imax, log=log, res=res, label="%s iteration counts" % mesh)

sys.exit(status + res.write())
