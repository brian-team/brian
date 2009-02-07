'''
Proof of concept - CUBA on a cluster with MPI

Slower than on a single processor but it works!
'''
from brian import *
import pypar
import time

class ReceiverThreshold(Threshold):
    '''
    Receives spikes from other processor
    '''
    def __init__(self,owner):
        self.owner=owner # processor owning the group

    def __call__(self,P):
        nspikes=array([0])
        pypar.broadcast(nspikes,self.owner)
        nspikes=nspikes[0]
        spikes=zeros(nspikes,dtype=int)
        if nspikes>0:
            pypar.broadcast(spikes,self.owner)
        return spikes

class SenderThreshold(Threshold):
    '''
    Sends spikes to other processors
    '''
    def __init__(self,owner,threshold):
        self.owner=owner # processor owning the group
        self._threshold=threshold

    def __call__(self,P):
        spikes=array(self._threshold(P),dtype=int)
        nspikes=array([len(spikes)])
        pypar.broadcast(nspikes,self.owner)
        nspikes=nspikes[0]
        if nspikes>0:
            pypar.broadcast(spikes,self.owner)
        return spikes

# Identification
myid =    pypar.rank() # id of this process
nproc = pypar.size() # number of processors
node =    pypar.get_processor_name()

print "I am processor %d of %d on node %s" %(myid, nproc, node)

if nproc!=2:
    raise Exception,"This example only works with 2 processors"

N=32000
Ne=int(N*.8)
Ni=int(N*.2)
sparseness=80./N
taum=20*ms
taue=5*ms
taui=10*ms
Vt=-50*mV
Vr=-60*mV
El=-49*mV
we=(60*0.27/10)*mV # excitatory synaptic weight (voltage)
wi=(-20*4.5/10)*mV # inhibitory synaptic weight

eqs= Equations('''
dv/dt  = (ge+gi-(v-El))/taum : volt
dge/dt = -ge/taue : volt
dgi/dt = -gi/taui : volt
''')

if myid==0: # excitatory group
    Pe=NeuronGroup(Ne,model=eqs,threshold=Vt,reset=Vr,refractory=5*ms)
    Pe._threshold=SenderThreshold(0,Pe._threshold)
    Pi=NeuronGroup(Ni,model=eqs,threshold=ReceiverThreshold(1))
    Pi._state_updater=LazyStateUpdater()
    Pe.v=Vr+rand(len(Pe))*(Vt-Vr)
    Cee=Connection(Pe,Pe,'ge',weight=we,sparseness=sparseness)
    Cie=Connection(Pi,Pe,'gi',weight=wi,sparseness=sparseness)
    
else: # inhibitory group
    Pi=NeuronGroup(Ni,model=eqs,threshold=Vt,reset=Vr,refractory=5*ms)
    Pi._threshold=SenderThreshold(1,Pi._threshold)
    Pe=NeuronGroup(Ne,model=eqs,threshold=ReceiverThreshold(0))
    Pe._state_updater=LazyStateUpdater()
    Pi.v=Vr+rand(len(Pi))*(Vt-Vr)
    Cei=Connection(Pe,Pi,'ge',weight=we,sparseness=sparseness)
    Cii=Connection(Pi,Pi,'gi',weight=wi,sparseness=sparseness)

# Record the number of spikes
Me=PopulationSpikeCounter(Pe)
Mi=PopulationSpikeCounter(Pi)
# A population rate monitor
#M = PopulationRateMonitor(P)

print len(Pe)+len(Pi),"neurons in the network"
print "Simulation running..."
start_time=time.time()

run(1*second)

duration=time.time()-start_time
print "Simulation time:",duration,"seconds"
print Me.nspikes,"excitatory spikes"
print Mi.nspikes,"inhibitory spikes"
#plot(M.times/ms,M.smooth_rate(2*ms,'gaussian'))
#show()

pypar.finalize()
