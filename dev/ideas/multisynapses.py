# -*- coding:utf-8 -*-
"""
Multiple synapses for each axon
"""
from brian import *

input=SpikeGeneratorGroup(2,[(0,10*ms),(1,50*ms)])
neurons=NeuronGroup(3,model='dv/dt=-v/(10*ms):1')
synapses=Connection(input,neurons,'v',delay=True,max_delay=10*ms)

M=StateMonitor(neurons,'v',record=True)

#synapses[0,0]=1
#synapses.delay[0,0]=2*ms

synapses.W.rows[0]=[1,1]
synapses.W.data[0]=[1,.5]
synapses.delayvec.rows[0]=[1,1]
synapses.delayvec.data[0]=[2*ms,5*ms]

run(80*ms)

plot(M.times/ms,M[0],'r')
plot(M.times/ms,M[1],'b')
plot(M.times/ms,M[2],'k')
show()
