from brian import *

G = NeuronGroup(1, 'v:1', threshold=-1, refractory=1*ms)

M = SpikeMonitor(G)

run(10*ms)

print M.spikes