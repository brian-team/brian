from brian import *
from brian.experimental.cuda.briantonemo import *

defaultclock.dt = 1*ms

layerfanout = 3   # each layer connects to this many other layers
layerdepth = 2    # length of chain of layers
numperlayer = 100 # number of neurons in each layer

tau = 10*ms
Vr = 0
Vt = 1
Vrest0 = 2 # resting potential of first layer (greater than Vt to force spikes)
delaymin, delaymax = 5*ms, 15*ms
weight = 0.02
layerp = 1.0 # probability of connection from layer to layer
refractory = 1*second

eqs = '''
dV/dt = -(V-Vrest)/tau : 1
Vrest : 1
'''

# Graph structure of the connections, each layer connects to three other layers
# and we store the summed delays between layers so that we can plot it
layers = {}
layerdelay = {0:0*second}
nextlayer = 1
curlayers = [0]
for i in xrange(layerdepth):
    nextcurlayers = []
    for layer in curlayers:
        for j in xrange(layerfanout):
            delay = rand()*(delaymax-delaymin)+delaymin
            layers[layer, nextlayer] = delay
            nextcurlayers.append(nextlayer)
            layerdelay[nextlayer] = delay+layerdelay[layer]
            nextlayer += 1
    curlayers = nextcurlayers

# Geometric series!
numlayers = (layerfanout**(layerdepth+1)-1)/(layerfanout-1)
N = numlayers*numperlayer

G = [NeuronGroup(numperlayer, eqs, reset=Vr, threshold=Vt, 
                 refractory=refractory) for _ in xrange(numlayers)]
for g in G:
    g.V = rand(numperlayer)*(Vt-Vr)+Vr
    g.Vrest = 0
# Initialise layer 0 to fire spikes
G[0].Vrest = Vrest0

C = {}
for (i, j), d in layers.items():
    # connect neurons in layer i to layer j with delay d
    C[i, j] = Connection(G[i], G[j], 'V', delay=(d, d), max_delay=delaymax,
                         sparseness=layerp, weight=weight, structure='sparse')

M = [SpikeMonitor(g) for g in G]

if 1:
    net = NemoNetwork(G, C.values(), M)
else:
    net = Network(G, C.values(), M)

net.run(.1*second)

print 'total spikes:', sum(m.nspikes for m in M)

if 1:
    raster_plot(*M)
    for i, d in layerdelay.items():
        fill([d/ms, (d+tau)/ms, (d+tau)/ms, d/ms],
             [i, i, i+1, i+1],
             color=(0.5, 0.5, 0.5))
    show()

