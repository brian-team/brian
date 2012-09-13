#!/usr/bin/env python
'''
Synfire chains
--------------
M. Diesmann et al. (1999). Stable propagation of synchronous spiking in cortical
neural networks. Nature 402, 529-533.
'''
from brian import *
# Neuron model parameters
Vr = -70 * mV
Vt = -55 * mV
taum = 10 * ms
taupsp = 0.325 * ms
weight = 4.86 * mV
# Neuron model
eqs = Equations('''
dV/dt=(-(V-Vr)+x)*(1./taum) : volt
dx/dt=(-x+y)*(1./taupsp) : volt
dy/dt=-y*(1./taupsp)+25.27*mV/ms+\
    (39.24*mV/ms**0.5)*xi : volt
''')
# Neuron groups
P = NeuronGroup(N=1000, model=eqs,
    threshold=Vt, reset=Vr, refractory=1 * ms)
Pinput = PulsePacket(t=50 * ms, n=85, sigma=1 * ms)
# The network structure
Pgp = [ P.subgroup(100) for i in range(10)]
C = Connection(P, P, 'y')
for i in range(9):
    C.connect_full(Pgp[i], Pgp[i + 1], weight)
Cinput = Connection(Pinput, Pgp[0], 'y')
Cinput.connect_full(weight=weight)
# Record the spikes
Mgp = [SpikeMonitor(p) for p in Pgp]
Minput = SpikeMonitor(Pinput)
monitors = [Minput] + Mgp
# Setup the network, and run it
P.V = Vr + rand(len(P)) * (Vt - Vr)
run(100 * ms)
# Plot result
raster_plot(showgrouplines=True, *monitors)
show()
