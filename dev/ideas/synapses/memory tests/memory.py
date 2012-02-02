"""
Memory consumption of the Synapses vs. Connection object

All-to-all connections
N=5000 -> 25e6 synapses

1054

Synapses:
1 variable:           611 MB -> 24B/syn
3 variables:          985 MB
3 vars + pre/post:    1037 MB -> 41B/syn

Connection (sparse but no delay):
1151 MB -> 46B/syn
DelayConnection (1 ms max delay)
1164 MB
DelayConnection + STDP (1 ms max delay)
1155 MB
"""
from brian import *
from time import time
from brian.experimental.synapses import *
#from guppy import hpy
#h = hpy()

N=5000

#print h.heap()

#1740 -> 574
P=NeuronGroup(N,model="v:1")
k=0
if k==0:
    S=Synapses(P,model='''w:1
                          Apre:1
                          Apost:1''',pre='v+=w',post='w+=Apre')
    S[:,:]=True
elif k==1:
    C=Connection(P,P,'v',delay=True,max_delay=1*ms)
    C.connect_full(P,P,weight=1.)
elif k==2:
    C=Connection(P,P,'v',delay=True,max_delay=1*ms)
    C.connect_full(P,P,weight=1.)
    S=ExponentialSTDP(C, 10*ms, 10*ms, 1., 1.,wmax=1.)

_=raw_input("Press enter")
#print h.heap()
