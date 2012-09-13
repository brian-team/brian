#!/usr/bin/env python
"""
Probabilistic synapses - Katz model
"""
from brian import *
from numpy.random import binomial
        
Nin=1000
Nout=25
input=PoissonGroup(Nin,rates=2*Hz)
tau=10*ms
neurons=NeuronGroup(Nout,model="dv/dt=-v/tau:1",threshold=35*50./5,reset=0)
S=Synapses(input,neurons,model='''w:1 # PSP size for one quantum
                                  nvesicles:1 # Number of vesicles (n is reserved)
                                  p:1 # Release probability''',
                         pre ='''v+=binomial(nvesicles,p)*w''')
S[:,:]=True # all-to-all
S.w='rand()'
S.nvesicles=50
S.p='rand()'

S=SpikeMonitor(neurons)

run(1000*ms)

raster_plot(S)
show()
