from brian import *

layerfanout = 3
layerdepth = 3
numperlayer = 100

tau = 10*ms
Vr = 0
Vt = 1
Vrest0 = 2
tauVrest = 200*ms
delaymin, delaymax = 5*ms, 15*ms
weight = 0.02
layerp = 1.0
refractory = 1*second

eqs = '''
dV/dt = -(V-Vrest)/tau : 1
dVrest/dt = -Vrest/tauVrest : 1
'''

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

numlayers = (layerfanout**(layerdepth+1)-1)/(layerfanout-1)
N = numlayers*numperlayer

G = NeuronGroup(N, eqs, reset=Vr, threshold=Vt, refractory=refractory)
G.V = rand(N)*(Vt-Vr)+Vr
G.Vrest = 0
G.Vrest[0:numperlayer] = Vrest0

C = Connection(G, G, 'V', delay=True, max_delay=delaymax)

for (i, j), d in layers.items():
    # connect neurons in layer i to layer j with delay d
    C.connect_random(G[i*numperlayer:(i+1)*numperlayer],
                     G[j*numperlayer:(j+1)*numperlayer],
                     p=layerp, delay=d, weight=weight)

M = SpikeMonitor(G)

run(.1*second)

raster_plot(M)
for i, d in layerdelay.items():
    fill([d/ms, (d+tau)/ms, (d+tau)/ms, d/ms],
         [i*numperlayer, i*numperlayer, (i+1)*numperlayer, (i+1)*numperlayer],
         color=(0.5, 0.5, 0.5))
show()
