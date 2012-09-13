#!/usr/bin/env python
'''
Topographic map - an example of complicated connections.
Two layers of neurons.
The first layer is connected randomly to the second one in a
topographical way.
The second layer has random lateral connections.
Each neuron has a position x[i].
'''
from brian import *

N = 100
tau = 10 * ms
tau_e = 2 * ms # AMPA synapse
eqs = '''
dv/dt=(I-v)/tau : volt
dI/dt=-I/tau_e : volt
'''

rates = zeros(N) * Hz
rates[N / 2 - 10:N / 2 + 10] = ones(20) * 30 * Hz
layer1 = PoissonGroup(N, rates=rates)
layer1.x = linspace(0., 1., len(layer1)) # abstract position between 0 and 1
layer2 = NeuronGroup(N, model=eqs, threshold=10 * mV, reset=0 * mV)
layer2.x = linspace(0., 1., len(layer2))

# Generic connectivity function
topomap = lambda i, j, x, y, sigma: exp(-abs(x[i] - y[j]) / sigma)

feedforward = Connection(layer1, layer2, sparseness=.5,
                           weight=lambda i, j:topomap(i, j, layer1.x, layer2.x, .3) * 3 * mV)

recurrent = Connection(layer2, layer2, sparseness=.5,
                         weight=lambda i, j:topomap(i, j, layer1.x, layer2.x, .2) * .5 * mV)

spikes = SpikeMonitor(layer2)

run(1 * second)
subplot(211)
raster_plot(spikes)
subplot(223)
imshow(feedforward.W.todense(), interpolation='nearest', origin='lower')
title('Feedforward connection strengths')
subplot(224)
imshow(recurrent.W.todense(), interpolation='nearest', origin='lower')
title('Recurrent connection strengths')
show()
