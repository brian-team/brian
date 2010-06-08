import numpy
import time
import random
from itertools import repeat

# units
second = 1.
ms = 0.001
mV = 0.001

# parameters
N = 4000
Ne = int(N * 0.8)
Ni = N - Ne
Nplot = 4
dt = 0.1 * ms
T = 200 * ms
numsteps = int(T / dt)
Vr = -60 * mV
Vt = -50 * mV
p = 0.02
we = 1.62 * mV
wi = -9 * mV

# Matrices for state update, S(t+dt)=A*S+_C
A = numpy.array([[ 0.99501248, 0.00493794, 0.00496265],
 [ 0.     , 0.98019867, 0.        ],
 [ 0.  , 0.  , 0.99004983]])
_C = numpy.array([[ -2.44388520e-04],
 [ -8.58745657e-21],
 [  6.90431479e-20]])

# Initialise state matrix and assign uniform random membrane potentials
S = numpy.zeros((3, N))
S[0, :] = [random.uniform(Vr, Vt) for _ in xrange(N)]

# Generate random connectivity matrix (note: no weights)
W = []
for _ in xrange(N):
    k = numpy.random.binomial(N, p, 1)[0]
    a = random.sample(xrange(N), k)
    a.sort()
    a = numpy.array(a)
    W.append(a)

Vrec = [[] for _ in range(Nplot)]
spikesrec = []

t = 0.
Nspikes = 0
start = time.time()
for _ in xrange(numsteps):
    S[:] = numpy.dot(A, S) + _C
    spikes = (S[0, :] > Vt).nonzero()[0]
    for i in spikes:
        if i < Ne:
            S[1, W[i]] += we
        else:
            S[2, W[i]] += wi
    S[0, spikes] = Vr
    Nspikes += len(spikes)
    spikesrec += zip(spikes, repeat(t))
    t += dt
    for i in range(Nplot):
        Vrec[i].append(S[0, i])

print 'Time taken', time.time() - start
print 'Spikes', Nspikes

try:
    import pylab
    sn, st = zip(*spikesrec)
    pylab.plot(st, sn, '.')
    pylab.figure()
    for i in range(Nplot):
        pylab.plot(Vrec[i])
    pylab.show()
except ImportError:
    print 'Cannot plot, no pylab'
