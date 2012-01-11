'''
One synapse
'''
from brian import *
from dev.ideas.synapses.synapses import *

log_level_debug()

P=NeuronGroup(1,model='dv/dt=1/(10*ms):1',threshold=1,reset=0)
Q=NeuronGroup(1,model='v:1')
S=Synapses(P,Q,model='w:1',pre='v+=w',max_delay=1*ms)
M=StateMonitor(Q,'v',record=True)

S.synapses_pre[0]=array([0],dtype=S.synapses_pre[0].dtype)
S.w[0]=1.
S.delay_pre[0]=5 # in timebins
S.presynaptic[0]=0
S.postsynaptic[0]=0

run(40*ms)

plot(M.times/ms,M[0])
show()
