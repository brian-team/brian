"""
Benchmark of a large network matching the specifications in
Morrison et al. (2005).
(Network from Brunel 2000).

Without synapses: 122s
With synapses: 220s

NEST: 1600-6400s depending on the machine

Interesting point: spike propagations are approximately as
expensive as state updates although the number of synapses is
huge. This is in fact not so surprising:

250 000 spike propagations / second
100 000 state updates / second
(and each state update involves random number generation)

This suggests that spike propagation is generally not a big issue
in practice.

Brian's simulation time might be an understimated because of
caching and/or memory access issues.
"""
from brian import *
from time import time

dt=defaultclock.dt

N=100000 # neurons
p=10000 # synapses (that's a billion synapses)
F=2.5*Hz # firing rate
duration=1*second

taum=20*ms
Vt=-50*mV
Vr=-60*mV
El=-49*mV
sigma=1*mV

eqs="""
dv/dt  = -(v-El)/taum + sigma/taum**.5*xi: volt
"""

p_active=F*dt
Nactive=int(N*p_active) # number of active neurons every time step
active_ones=arange(N)<Nactive
P=NeuronGroup(N,model=eqs,threshold="rand(N)<p_active",reset=Vr,refractory=2*ms,delay=1*ms)

# We design a fake connection matrix that always returns the
# same row (thus the simulation time is unchanged but there is no
# memory constraint).
class FakeConstructionMatrix(SparseConstructionMatrix):
    def connection_matrix(self):
        return FakeConnectionMatrix(self, **self.init_kwds)
class FakeConnectionMatrix(SparseConnectionMatrix):
    def get_row(self, i):
        return self.rows[0]
    def get_rows(self, rows):
        return [self.rows[0] for i in rows]

C=Connection(P,P,'v',structure=FakeConstructionMatrix)
C.connect_random(P[:1],P,sparseness=p*1./N,weight=1*mV)

M=SpikeMonitor(P,record=False)

print "Starting..."
t1=time()
run(duration)
t2=time()
print "It took",t2-t1,"s"
print M.nspikes,"spikes"
