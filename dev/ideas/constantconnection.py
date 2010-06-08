from brian import *

from random import sample
from scipy import random
from brian.connection import ConnectionMatrix


class ConstantSparseConnectionMatrix(ConnectionMatrix):
    def __init__(self, dims, val):
        self.sourcelen, self.targetlen = dims
        self.rows = [ array([], dtype=int) for _ in xrange(self.sourcelen) ]
        self.val = val

    def add_row(self, i, X):
        X[self.rows[i]] += self.val

    def add_scaled_row(self, i, X, factor):
        # modulation may not work? need factor[self.rows[i]] here? is factor a number or an array?
        X[self.rows[i]] += factor * self.val

    def random_matrix(self, i_start, i_end, m, offset, p):
        for i in xrange(i_start, i_end):
            k = random.binomial(m, p, 1)[0]
            r = offset + array(sample(xrange(m), k), dtype=int)
            self.rows[i] = r


class ConstantConnection(Connection):
    def __init__(self, source, target, state=0, delay=0 * msecond, modulation=None,
                 weight=1.):
        Connection.__init__(self, source, target, state=state, delay=delay, modulation=modulation)
        self.W = ConstantSparseConnectionMatrix((len(source), len(target)), weight)
        self.weight = weight

    def connect_random(self, P, Q, p):
        weight = self.weight
        try:
            weight + Q._S0[self.nstate]
        except DimensionMismatchError, inst:
            raise DimensionMismatchError("Incorrects unit for the synaptic weights.", *inst._dims)
        i_start = P._origin - self.source._origin
        i_end = i_start + len(P)
        offset = Q._origin - self.target._origin
        m = len(Q)
        self.W.random_matrix(i_start, i_end, m, offset, p)


eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

N = 64000
duration = 500 * ms
usecc = True

Ne = int(0.8 * N)
Ni = N - Ne

P = NeuronGroup(N, model=eqs,
              threshold= -50 * mV, reset= -60 * mV)
P.v = -60 * mV + 10 * mV * rand(len(P))
Pe = P.subgroup(Ne)
Pi = P.subgroup(Ni)

if usecc:
    Ce = ConstantConnection(Pe, P, 'ge', weight=1.62 * mV)
    Ci = ConstantConnection(Pi, P, 'gi', weight= -9 * mV)
    Ce.connect_random(Pe, P, 0.02)
    Ci.connect_random(Pi, P, 0.02)
else:
    Ce = Connection(Pe, P, 'ge')
    Ci = Connection(Pi, P, 'gi')
    Ce.connect_random(Pe, P, 0.02, weight=1.62 * mV)
    Ci.connect_random(Pi, P, 0.02, weight= -9 * mV)

#M=SpikeMonitor(P)
M = PopulationSpikeCounter(P)

import time
t = time.time()
run(duration)
print time.time() - t
#raster_plot(M)
#show()
print M.nspikes
