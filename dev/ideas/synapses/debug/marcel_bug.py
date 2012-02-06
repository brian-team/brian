from brian import *
from brian.experimental.synapses import *

n1, n2 = 10, 50
input = SpikeGeneratorGroup(n1, [(i, 10*ms) for i in xrange(n1)])
G = NeuronGroup(n1 * n2, model='dv/dt = -v/(5*ms) : 1', threshold=1, reset=0)

S = Synapses(input, G, model='w : 1', pre='v += w', max_delay=5*ms)
for i in xrange(n1):
    for j in xrange(n2):
        S[i, j*n1 + i] = 2
        S.w[i, j*n1 + i] = [0.1, 0.1]
        S.delay[i, j*n1 + i] = [0 * ms, 5 * ms]

counter = SpikeCounter(G)
v_mon = StateMonitor(G, 'v', record=0)
run(1 * second, report='text')

print counter.count[:n1]
figure()
v_mon.plot()
show()
