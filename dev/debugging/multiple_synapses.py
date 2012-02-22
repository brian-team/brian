from brian import *

from brian.experimental.synapses import Synapses

G1 = SpikeGeneratorGroup(1, [(0, 0*ms)])
G2 = NeuronGroup(2, model='dv/dt = -v/(0.5*ms) : 1')

S = Synapses(G1, G2, model='w:1', pre='v += w')
S[:, :] = 2
# Using this instead works correctly
#S[0, 0] = 2
#S[0, 1] = 2

S.w[0, 0] = [0.1, 0.2]
S.w[0, 1] = [0.3, 0.4]
S.delay[0, 0] = [0*ms, 1*ms]
S.delay[0, 1] = [2*ms, 3*ms]

mon = StateMonitor(G2, 'v', record=True)
run(10*ms)

plot(mon.times/ms, mon.values.T)
show()
