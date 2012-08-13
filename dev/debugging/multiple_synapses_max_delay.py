from brian import *

inp = SpikeGeneratorGroup(1, [(0,0*ms),(0, 1 * ms)])
G = NeuronGroup(1, model='v:1')
syn = Synapses(inp, G, model='w:1', pre='v+=w', max_delay=5*ms)
mon = StateMonitor(G, 'v', record=True)

syn[:, :] = 2
syn.w[:, :] = 1
syn.delay[:, :] = [1 * ms, 1 * ms]
net = Network(inp, G, syn, mon)
net.run(defaultclock.dt)

#print syn.queues[0].

syn.delay[:, :] = [7 * ms, 5 * ms]

net.run(6.5 * ms)

mon.plot()
show()


