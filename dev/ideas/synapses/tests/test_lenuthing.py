from brian import *
from brian.experimental.synapses import *

reinit_default_clock()

source = SpikeGeneratorGroup(3, [(0, 1*ms), (1, 1*ms), (2, 1*ms)])
target = NeuronGroup(2, model = '''dv/dt = -v/(10*ms):1''')

syn = Synapses(source, target, 
               model = 'w:1', pre = 'v+=w')

syn[1,1] = 1
syn[2,1] = 1.
syn[0,0] = 1
syn[1,0] = 1
syn[2,0] = 1

syn.w = 50

M = StateMonitor(target, 'v', record = True)
print syn
print syn.w[:],syn.presynaptic,syn.postsynaptic

run(40*ms)



plot(M.times/ms, M[0])
plot(M.times/ms, M[1])
show()



