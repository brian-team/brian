from brian import *

N = 100

H = SpikeGeneratorGroup(1, [(0, 0*ms)])
G = NeuronGroup(N, 'V:1', threshold=1, reset=0)

C = Connection(H, G, structure='dense', delay=True, weight=2.0, max_delay=10*ms)
C.delay[0, :] = linspace(0*ms, 10*ms, N)

sp = SpikeMonitor(G)
Mv = StateMonitor(G, 'V', record=True)

run(15*ms)

subplot(211)
raster_plot(sp)
subplot(212)
for i in range(N):
    plot(Mv.times, Mv[i]+i)
show()