from brian import *
from brian.experimental.synapses import *

P=NeuronGroup(1,model='dv/dt=1/(10*ms):1',threshold=1,reset=0)
Q=NeuronGroup(1,model='v:1')
S=Synapses(P,Q,model='w:1',pre='v+=w',max_delay=1*ms)
