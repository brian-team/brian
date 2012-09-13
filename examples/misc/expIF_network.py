#!/usr/bin/env python
'''
A network of exponential IF models with synaptic conductances
'''
from brian import *
from brian.library.IF import *
from brian.library.synapses import *
import time

C = 200 * pF
taum = 10 * msecond
gL = C / taum
EL = -70 * mV
VT = -55 * mV
DeltaT = 3 * mV

# Synapse parameters
Ee = 0 * mvolt
Ei = -80 * mvolt
taue = 5 * msecond
taui = 10 * msecond

eqs = exp_IF(C, gL, EL, VT, DeltaT)
# Two different ways of adding synaptic currents:
eqs += Current('''
Ie=ge*(Ee-vm) : amp
dge/dt=-ge/taue : siemens
''')
eqs += exp_conductance('gi', Ei, taui) # from library.synapses

P = NeuronGroup(4000, model=eqs, threshold= -20 * mvolt, reset=EL, refractory=2 * ms)
Pe = P.subgroup(3200)
Pi = P.subgroup(800)
we = 1.5 * nS # excitatory synaptic weight
wi = 2.5 * we # inhibitory synaptic weight
Ce = Connection(Pe, P, 'ge', weight=we, sparseness=0.05)
Ci = Connection(Pi, P, 'gi', weight=wi, sparseness=0.05)
# Initialization
P.vm = randn(len(P)) * 10 * mV - 70 * mV
P.ge = (randn(len(P)) * 2 + 5) * we
P.gi = (randn(len(P)) * 2 + 5) * wi

# Excitatory input to a subset of excitatory and inhibitory neurons
# Excitatory neurons are excited for the first 200 ms
# Inhibitory neurons are excited for the first 100 ms
input_layer1 = Pe.subgroup(200)
input_layer2 = Pi.subgroup(200)
input1 = PoissonGroup(200, rates=lambda t: (t < 200 * ms and 2000 * Hz) or 0 * Hz)
input2 = PoissonGroup(200, rates=lambda t: (t < 100 * ms and 2000 * Hz) or 0 * Hz)
input_co1 = IdentityConnection(input1, input_layer1, 'ge', weight=we)
input_co2 = IdentityConnection(input2, input_layer2, 'ge', weight=we)

# Record the number of spikes
M = SpikeMonitor(P)

print "Simulation running..."
start_time = time.time()
run(500 * ms)
duration = time.time() - start_time
print "Simulation time:", duration, "seconds"
print M.nspikes / 4000., "spikes per neuron"
raster_plot(M)
show()
