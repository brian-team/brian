#!/usr/bin/env python
'''
Network to demonstrate stopping a simulation during a run

Have a fully connected network of integrate and fire neurons
with input fed by a group of Poisson neurons with a steadily
increasing rate, want to determine the point in time at which
the network of integrate and fire neurons switches from no
firing to all neurons firing, so we have a network_operation
called stop_condition that calls the stop() function if the
monitored network firing rate is above a minimum threshold.
'''

from brian import *

clk = Clock()

Vr = 0 * mV
El = 0 * mV
Vt = 10 * mV
tau = 10 * ms
weight = 0.2 * mV
duration = 100 * msecond
max_input_rate = 10000 * Hz
num_input_neurons = 1000
input_connection_p = 0.1
rate_per_neuron = max_input_rate / (num_input_neurons * input_connection_p)

P = PoissonGroup(num_input_neurons, lambda t: rate_per_neuron * (t / duration))

G = NeuronGroup(1000, model='dV/dt=-(V-El)/tau : volt', threshold=Vt, reset=Vr)
G.V = Vr + (Vt - Vr) * rand(len(G))

CPG = Connection(P, G, weight=weight, sparseness=input_connection_p)

CGG = Connection(G, G, weight=weight)

MP = PopulationRateMonitor(G, bin=1 * ms)

@network_operation
def stop_condition():
    if MP.rate[-1] * Hz > 10 * Hz:
        stop()

run(duration)

print "Reached population rate>10 Hz by time", clk.t, "+/- 1 ms."
