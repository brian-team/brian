from brian import *
from dev.ideas.synapses.synapses import *

log_level_debug()

P=NeuronGroup(1,model='dv/dt=1/(10*ms):1',threshold=1,reset=0)
Q=NeuronGroup(1,model='v:1')
S=Synapses(P,Q,model='w:1',pre='w=1',max_delay=1*ms)
S[0,0]=True

run(40*ms)

print S.w[:]
