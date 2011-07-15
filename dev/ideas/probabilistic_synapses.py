"""
Stochastic synaptic transmission.

In this example, two Poisson neurons project to N=20 neurons.
For the first input, the transmission probability increases (0 to 1)
from neuron 0 to neuron 19. 
For the second input, the transmission probability decreases (1 to 0)
from neuron 0 to neuron 19.
The membrane potential of all 20 neurons are shown.
"""
from brian import *

# This is dense, but we could do a sparse one
class ProbabilisticConstructionMatrix(DenseConstructionMatrix):
    def __init__(self,shape,**kwds):
        DenseConstructionMatrix.__init__(self,shape,**kwds)
        self.w0=1. # constant synaptic weight

    def connection_matrix(self):
        return ProbabilisticConnectionMatrix(self)

class ProbabilisticConnectionMatrix(DenseConnectionMatrix):
    def __init__(self, val, **kwds):
        DenseConnectionMatrix.__init__(self, val, **kwds)
        self.w0=val.w0

    def get_rows(self, rows):
        return [(rand(len(self.rows[i]))<self.rows[i])*self.w0 for i in rows]

N=20
tau=5*ms
input=PoissonGroup(2,rates=20*Hz)
neurons=NeuronGroup(N,model='dv/dt=-v/tau : 1')
synapses=Connection(input,neurons,structure=ProbabilisticConstructionMatrix)
synapses.W.w0=0.5 # EPSP size
# Transmission probabilities
synapses.W[0,:]=linspace(0,1,N) # transmission probability between 0 and 1
synapses.W[1,:]=linspace(0,1,N)[::-1] # reverse order for the second input

M=StateMonitor(neurons,'v',record=True)

run(500*ms)

for i in range(N):
    plot(M.times/ms,M[i]+i,'k')
show()
