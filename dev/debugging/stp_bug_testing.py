from brian import *

G = NeuronGroup(1, 'dV/dt=rate:1\nrate:Hz', threshold=1, reset=0)
G.rate = 20*Hz
G.m = SpikeMonitor(G)
H = NeuronGroup(1, 'dV/dt=-V/ms:1')
M2 = StateMonitor(H, 'V', record=True)
C = Connection(G, H, 'V', weight=1)
stp = STP(C, taud=0.1*ms, tauf=5000*ms, U=0.2)
M = MultiStateMonitor(stp.vars, record=True)
run(.5*second)
G.rate = 0*Hz # try it also commenting out this line!
run(.5*second)
G.rate = 20*Hz
run(.5*second)

subplot(211)
M.plot()
legend()
subplot(212)
M2.plot()
show()