#!/usr/bin/env python
'''
This example shows how to efficiently simulate neurons with a large number of 
Poisson inputs targetting arbitrarily complex synapses. The approach is very
similiar to what the :class:`PoissonInput` class does internally, but
:class:`PoissonInput` cannot be combined with the :class:`Synapses` class.
You could also just use many :class:`PoissonGroup` objects as inputs, but this
is very slow and memory consuming.    
'''
from brian import *

# Poisson inputs
M = 1000 # number of Poisson inputs
max_rate = 100

# Neurons
N = 50 # number of neurons
tau = 10 * ms
E_exc = 0 * mV
E_L = -70 * mV
G = NeuronGroup(N, model='dvm/dt = -(vm - E_L)/tau : mV')
G.rest()

# Dummy neuron group
P = NeuronGroup(1, 'v : 1', threshold= -1, reset=0) # spikes every timestep

# time varying rate
def varying_rate(t):
    return defaultclock.dt * max_rate * (0.5 + 0.5 * sin(2 * pi * 5 * t))

# Synaptic connections: binomial(cellM, varying_rate(t)) gives the number of
# events per timestep. The synapse model is a conductance-based instanteneous
# jump in postsynaptic membrane potential 
S = Synapses(P, G, model='''
                            J : 1
                            cellM : 1 
                        ''',
             pre='vm += binomial(cellM, varying_rate(t)) * J * (E_exc - vm)')
S[:, :] = True
S.cellM = M #we need one value for M per cell, so that binomial is vectorized
S.J = 0.0005
    
mon = StateMonitor(G, 'vm', record=True)
run(1 * second, report='text')

mon.plot()
show()