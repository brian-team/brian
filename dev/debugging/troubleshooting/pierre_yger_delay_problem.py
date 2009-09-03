# SOLUTION 2: DERIVED CLASS

from brian import *
from brian.directcontrol import MultipleSpikeGeneratorThreshold

class MultipleSpikeGeneratorGroupWithDelays(MultipleSpikeGeneratorGroup):
    def __init__(self,spiketimes,clock=None,max_delay=0*ms):
        clock = guess_clock(clock)
        thresh = MultipleSpikeGeneratorThreshold(spiketimes)
        NeuronGroup.__init__(self,len(spiketimes),model=LazyStateUpdater(),threshold=thresh,clock=clock,max_delay=max_delay)

tau=10*ms
sigma=5*mV
eqs='dv/dt = -v/tau : volt'
pop=NeuronGroup(1,model=eqs,threshold=10*mV,reset=0*mV,refractory=5*ms)
input=MultipleSpikeGeneratorGroupWithDelays([[0.1*second,0.2*second,0.3*second,0.4*second,0.5*second]],max_delay=3*ms)
pop.v=-60*mV
C=Connection(input,pop,'v',delay=2*ms)
C.connect_one_to_one(input, pop, weight=15*mV)
Minp = SpikeMonitor(input)
M=SpikeMonitor(pop,True)
run(1*second)
print Minp.spikes
print M.spikes


# SOLUTION 1: INTERMEDIATE GROUP
#
#from brian import *
#tau=10*ms
#sigma=5*mV
#eqs='dv/dt = -v/tau : volt'
#pop=NeuronGroup(1,model=eqs,threshold=10*mV,reset=0*mV,refractory=5*ms)
#inputgen=MultipleSpikeGeneratorGroup([[0.1*second,0.2*second,0.3*second,0.4*second,0.5*second]])
#input=NeuronGroup(len(inputgen),model='v:1',threshold=0.5,reset=0.,max_delay=3*ms)
#C_inputgen_input=IdentityConnection(inputgen,input)
#pop.v=-60*mV
#C=Connection(input,pop,'v',delay=2*ms-defaultclock.dt)
#C.connect_one_to_one(input, pop, weight=15*mV)
#Minpgen = SpikeMonitor(inputgen)
#Minp = SpikeMonitor(input)
#M=SpikeMonitor(pop,True)
#run(1*second)
#print Minpgen.spikes
#print Minp.spikes
#print M.spikes
