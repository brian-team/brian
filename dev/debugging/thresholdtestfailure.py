'''
Created on 4 sept. 2009

@author: goodman
'''

from brian import *

G = NeuronGroup(3,
                'v:1\nw:1',
                reset=Reset(0.,state=1),
                threshold=Threshold(1.,state=1),
                )

M = SpikeMonitor(G,True)

G.state(1)[:]=array([0.5,1.5,2.5])

run(1*msecond)

print M.spikes
print G._S
G._resetfun.statevectors[id(G)][:] = 5
print G._S
print G._resetfun.state