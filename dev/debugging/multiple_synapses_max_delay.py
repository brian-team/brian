from brian import *

inp = SpikeGeneratorGroup(1, [(0, 1 * ms)])
G = NeuronGroup(1, model='v:1')
syn = Synapses(inp, G, model='w:1', pre='v+=w', max_delay=5*ms)
mon = StateMonitor(G, 'v', record=True)

syn[:, :] = 2
syn.w[:, :] = 1
syn.delay[:, :] = [0 * ms, 0 * ms]

run(defaultclock.dt)

syn.delay[:, :] = [2.5 * ms, 5 * ms]

run(6 * ms)

mon.plot()
show()


