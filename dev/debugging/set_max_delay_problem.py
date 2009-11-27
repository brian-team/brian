from brian import *

G = NeuronGroup(100, 'dV/dt=(1.1-V)/(10*ms):1', reset=0, threshold=1)
H = G.subgroup(50)
C = Connection(H, G, 'V', weight=.3, sparseness=0.1)
M = SpikeMonitor(G)

G.V = rand(len(G))

run(100*ms)

G.LS.S = None
G.set_max_delay(G._max_delay*defaultclock.dt)

run(100*ms)

raster_plot(M)
show()