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

G = NeuronGroup(N, eqs, reset=Vr, threshold=Vt, refractory=refractory)
G.V = rand(N)*(Vt-Vr)+Vr
G.Vrest = 0
# Initialise layer 0 to fire spikes
G.Vrest[0:numperlayer] = Vrest0

C = NemoConnection(G, G, 'V', delay=True, max_delay=delaymax)

for (i, j), d in layers.items():
    # connect neurons in layer i to layer j with delay d
    C.connect_random(G[i*numperlayer:(i+1)*numperlayer],
                     G[j*numperlayer:(j+1)*numperlayer],
                     p=layerp, delay=d, weight=weight)

M = SpikeMonitor(G)

run(.1*second)

print 'total spikes:', M.nspikes

if 1:
    raster_plot(M)
    for i, d in layerdelay.items():
        fill([d/ms, (d+tau)/ms, (d+tau)/ms, d/ms],
             [i*numperlayer, i*numperlayer, (i+1)*numperlayer, (i+1)*numperlayer],
             color=(0.5, 0.5, 0.5))
    show()

