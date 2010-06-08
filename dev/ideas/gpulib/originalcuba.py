'''
Very short example program.
'''
from brian import *
import time

Nplot = 4
N = 4000
Ne = int(0.8 * N)
Ni = N - Ne
T = 200 * ms

eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

P = NeuronGroup(N, model=eqs,
              threshold= -50 * mV, reset= -60 * mV)
P.v = -60 * mV + 10 * mV * rand(len(P))
Pe = P.subgroup(Ne)
Pi = P.subgroup(Ni)

Ce = Connection(Pe, P, 'ge')
Ci = Connection(Pi, P, 'gi')
Ce.connect_random(Pe, P, 0.02, weight=1.62 * mV)
Ci.connect_random(Pi, P, 0.02, weight= -9 * mV)

M = SpikeMonitor(P)
MV = StateMonitor(P, 'v', record=range(Nplot))

start = time.time()
run(T)

print 'Time taken', time.time() - start
print 'Spikes', M.nspikes

raster_plot(M)
figure()
for i in range(Nplot):
    plot(MV[i])
show()
