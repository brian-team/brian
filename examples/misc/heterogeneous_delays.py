#!/usr/bin/env python
'''
Script demonstrating use of a :class:`Connection` with homogenenous delays

The network consists of a 'starter' neuron which fires a single spike at time
t=0, connected to 100 leaky integrate and fire neurons with different delays
for each target neuron, with the delays forming a quadratic curve centred at
neuron 50. The longest delay is 10ms, and the network is run for 40ms. At
the end, the delays are plotted above a colour plot of the membrane potential
of each of the target neurons as a function of time (demonstrating the
delays).
'''
from brian import *
# Starter neuron, threshold is below 0 so it fires immediately, reset is below
# threshold so it fires only once.
G = NeuronGroup(1, model='V:1', threshold= -1.0, reset= -2.0)
# 100 LIF neurons, no reset or threshold so they will not spike
H = NeuronGroup(100, model='dV/dt=-V/(10*ms):volt')
# Connection with delays, here the delays are specified as a function of (i,j)
# giving the delay from neuron i to neuron j. In this case there is only one
# presynaptic neuron so i will be 0.
C = Connection(G, H, weight=5 * mV, max_delay=10 * ms,
               delay=lambda i, j:10 * ms * (j / 50. - 1) ** 2)
M = StateMonitor(H, 'V', record=True)
run(40 * ms)
subplot(211)
# These are the delays from neuron 0 to neuron i in ms
plot([C.delay[0, i] / ms for i in range(100)])
ylabel('Delay (ms)')
title('Delays')
subplot(212)
# M.values is an array of all the recorded values, here transposed to make
# it fit with the plot above.
imshow(M.values.T, aspect='auto', extent=(0, 100, 40, 0))
xlabel('Neuron number')
ylabel('Time (ms)')
title('Potential')
show()
