import random as pyrandom

from brian import *

from scipy import random as scirandom
from brian.connection import ConnectionMatrix

class ConstantComputedRandomConnectionMatrix(ConnectionMatrix):
    def __init__(self, dims, val):
        self.sourcelen, self.targetlen = dims
        self.initseed = pyrandom.randint(100000,1000000) # replace this
        self.val = val
    def get_random_indices(self,i):
        pyrandom.seed(self.initseed+int(i))
        scirandom.seed(self.initseed+int(i))
        k = scirandom.binomial(self.m, self.p,1)[0]
        return self.offset + array(pyrandom.sample(xrange(self.m),k),dtype=int)
    def add_row(self,i,X):
        X[self.get_random_indices(i)]+=self.val
    def add_scaled_row(self,i,X,factor):
        # modulation may not work? need factor[self.rows[i]] here? is factor a number or an array?
        X[self.get_random_indices(i)]+=factor*self.val
    def random_matrix(self,i_start,i_end,m,offset,p):
        self.m = m
        self.offset = offset
        self.p = p

class ConstantComputedRandomConnection(Connection):
    def __init__(self,source,target,state=0,delay=0*msecond,modulation=None,
                 weight=1., p=0.1):
        Connection.__init__(self, source, target, state=state, delay=delay, modulation=modulation)
        self.W = ConstantComputedRandomConnectionMatrix((len(source),len(target)),weight)
        self.weight = weight
        self.connect_random(source,target,p)
    def connect_random(self,P,Q,p):
        weight = self.weight
        try:
            weight+Q._S0[self.nstate]
        except DimensionMismatchError,inst:
            raise DimensionMismatchError("Incorrects unit for the synaptic weights.",*inst._dims)
        i_start = P._origin - self.source._origin
        i_end = i_start + len(P)
        offset = Q._origin - self.target._origin
        m = len(Q)
        self.W.random_matrix(i_start, i_end, m, offset, p)


eqs='''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

N = 1000000#it works!
duration = 10*ms
usecc = True

Ne = int(0.8*N)
Ni = N-Ne

P=NeuronGroup(N,model=eqs,
              threshold=-50*mV,reset=-60*mV)
P.v=-60*mV+10*mV*rand(len(P))
Pe=P.subgroup(Ne)
Pi=P.subgroup(Ni)

if usecc:
    Ce=ConstantComputedRandomConnection(Pe,P,'ge',weight=1.62*mV,p=0.02)
    Ci=ConstantComputedRandomConnection(Pi,P,'gi',weight=-9*mV,p=0.02)
else:
    Ce=Connection(Pe,P,'ge')
    Ci=Connection(Pi,P,'gi')
    Ce.connect_random(Pe, P, 0.02, weight=1.62*mV)
    Ci.connect_random(Pi, P, 0.02, weight=-9*mV)

#M=SpikeMonitor(P)
M=PopulationSpikeCounter(P)

import time
t=time.time()
run(duration)
print time.time()-t
#raster_plot(M)
#show()
print M.nspikes