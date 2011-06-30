from brian import *
from scipy.sparse import csr_matrix
import time

N = 1000
p = 0.1
repeats = 1000

G = NeuronGroup(N, 'V:1')
H = NeuronGroup(N, 'V:1')
Cs = Connection(G, H, 'V', sparseness=p, weight=1.0, structure='sparse')
Cd = Connection(G, H, 'V', sparseness=p, weight=1.0, structure='dense')
Cs.compress()
Ws = Cs.W
Wd = Cd.W

VG = asarray(G.V)
VH = asarray(H.V)

rowj, rowdata = Ws.rowj, Ws.rowdata

def f():
    VG[:] = 0
    VG[:] = 0
    for i in arange(0, N): # i is the source
        VG[rowj[i]] += VH[i]*rowdata[i]
start = time.time()
for _ in xrange(repeats):
    f()
print 'Time using loop:', time.time()-start

def f():
    VG[:] = 0
    VG[:] = 0
    for i in xrange(N): # i is the source
        VG[rowj[i]] += VH[i]*rowdata[i]
start = time.time()
for _ in xrange(repeats):
    f()
print 'Time using optimised loop:', time.time()-start

def f():
    VG[:] = dot(VH, Wd)
start = time.time()
for _ in xrange(repeats):
    f()
print 'Time using dense:', time.time()-start

Wss = csr_matrix((Ws.alldata, Ws.allj, Ws.rowind), (N, N))
def f():
    VG[:] = Wss*VH
start = time.time()
for _ in xrange(repeats):
    f()
print 'Time using scipy sparse:', time.time()-start
