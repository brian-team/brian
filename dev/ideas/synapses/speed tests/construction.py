"""
Speed test for construction
"""
from brian import *
from time import time
from brian.experimental.synapses import *

N=1000
neurons=NeuronGroup(N,model='dv/dt=1/(10*ms):1', threshold=1, reset=0)
S=Synapses(neurons,model='w:1')

S[:,:]=0.2
t1=time()
for i in range(1000):
    S.w[i,10:30]=1.
t2=time()
print t2-t1
