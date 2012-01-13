'''
Topographic map - an example of complicated connections.
Two layers of neurons.
The first layer is connected randomly to the second one in a
topographical way.
The second layer has random lateral connections.
'''
from brian import *
from numpy.random import seed
seed(324328740)
from random import seed
seed(352347323)
from brian.experimental.cuda.briantonemo import *
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
layer2 = NeuronGroup(N, model=eqs, threshold=10 * mV, reset=0 * mV)

topomap = lambda i, j:exp(-abs(i - j) * .1) * 3 * mV
feedforward = Connection(layer1, layer2, sparseness=.5, weight=topomap)
#feedforward[2,3]=1*mV

lateralmap = lambda i, j:exp(-abs(i - j) * .05) * 0.5 * mV
recurrent = Connection(layer2, layer2, sparseness=.5, weight=lateralmap)

spikes = SpikeMonitor(layer2)
M = StateMonitor(layer2, 'v', record=[50,51,52])

net = MagicNetwork()
net.run(.1 * second)
#
#print len(feedforward.W.alldata), len(recurrent.W.alldata)
#print feedforward.W[50,:].ind
#if hasattr(net, 'nemo_sim'):
#    def print_tgts(n):
#        L = net.nemo_net.get_synapses_from(n)
#        L = [net.nemo_net.get_synapse_target(l) for l in L]
#        print L
#    print_tgts(50)
#    print_tgts(150)
#    print_tgts(250)

subplot(211)
raster_plot(spikes)
subplot(212)
M.plot()
show()

