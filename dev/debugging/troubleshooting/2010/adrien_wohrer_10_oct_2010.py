from brian import *

dt = 0.1*ms
N = 500

G = NeuronGroup(1, 'V:1')

G.V = TimedArray(linspace(0, 1, N), dt=dt)
M = StateMonitor(G, 'V', record=True)

net = MagicNetwork()
net.prepare()
try:
    print net.clocks
except:
    print net.clock, net.clock.dt

net.run(N*dt)

V = M[0]

subplot(211)
plot(V)
subplot(212)
plot(diff(V))
show()
