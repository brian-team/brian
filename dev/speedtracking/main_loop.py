'''
Main loop: 1.6 s / 100 s (= 1e6 time steps)
With a neuron group: 12.5 s
'''

from time import time
from brian import *

#N=NeuronGroup(1000,'v:volt')
run(1*ms)
t1=time()
run(100*second)
t2=time()
print t2-t1
