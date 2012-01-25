"""
Vectorised construction
"""
from brian import *
from time import time
from brian.experimental.synapses import *

N=2000

P=NeuronGroup(N,model="v:1")
S=Synapses(P,model='w:1')
t1=time()
S[:,:]='rand()'
t2=time()

print t2-t1,'s'
