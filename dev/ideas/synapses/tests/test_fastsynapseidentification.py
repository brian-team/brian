from numpy import *
import time

spikes = arange(10000) # presynaptic neurons that spiked
pre = ones(100) # presyn neuron per synapse
pre[50:] = arange(50)
pre[0] = 1


'''
Spec:

function(spikes, pre) should return the array of synapse numbers concerned by the spikes in spikes (a list)

Issues:
- multiple synapse per same presyn neuron
- speed
'''

def dummy_method(s, p):
    synapses = []
    for i in s:
        idx = list(nonzero(p == i)[0])
        synapses+=idx
    return synapses

if True:
    t0 = time.time()
    resdummy = dummy_method(spikes, pre)
    print 'dummy method in ', time.time()-t0


def meshgrid_method(s, p):
    ss, pp = meshgrid(s, p)
    synapses = nonzero(sum(ss == pp, axis = 1))[0]
    return synapses

if True:
    t0 = time.time()
    resmesh = meshgrid_method(spikes, pre)
    print 'mesh method in ', time.time()-t0

print (sort(resmesh) == sort(resdummy)).all()

