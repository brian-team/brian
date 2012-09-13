#!/usr/bin/env python
'''
This example demonstrates using different clocks for different objects
in the network. The clock ``simclock`` is the clock used for the
underlying simulation. The clock ``monclock`` is the clock used for
monitoring the membrane potential. This monitoring takes place less
frequently than the simulation update step to save time and memory.
Finally, the clock ``inputclock`` controls when the external 'current'
``Iext`` should be updated. In this  case, we update it infrequently
so we can see the effect on the network.

This example also demonstrates the @network_operation decorator. A
function with this decorator will be run as part of the network
update step, in sync with the clock provided (or the default one
if none is provided).
'''

from brian import *
# define the three clocks
simclock = Clock(dt=0.1 * ms)
monclock = Clock(dt=0.3 * ms)
inputclock = Clock(dt=100 * ms)
# simple leaky I&F model with external 'current' Iext as a parameter
tau = 10 * ms
eqs = '''
dV/dt = (-V+Iext)/tau : volt
Iext: volt
'''
# A single leaky I&F neuron with simclock as its clock
G = NeuronGroup(1, model=eqs, reset=0 * mV, threshold=10 * mV, clock=simclock)
G.V = 5 * mV
# This function will be run in sync with inputclock i.e. every 100 ms
@network_operation(clock=inputclock)
def update_Iext():
    G.Iext = rand(len(G)) * 20 * mV
# V is monitored in sync with monclock
MV = StateMonitor(G, 'V', record=0, clock=monclock)
# run and plot 
run(1000 * ms)
plot(MV.times / ms, MV[0] / mV)
show()
# You should see 10 different regions, sometimes Iext will be above threshold
# in which case you will see regular spiking at different rates, and sometimes
# it will be below threshold in which case you'll see exponential decay to that
# value
