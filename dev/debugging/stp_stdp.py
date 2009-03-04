from brian import *

G = NeuronGroup(1, 'dV/dt=rate:1\nrate:Hz', threshold=1, reset=0)
G.rate = 20*Hz
G.m = SpikeMonitor(G)
H = NeuronGroup(1, 'dV/dt=-V/ms:1', threshold=0.5, reset=0)
M2 = StateMonitor(H, 'V', record=True)
C = Connection(G, H, 'V', weight=1)

stp = STP(C, taud=0.1*ms, tauf=5000*ms, U=0.2)
stdp = ExponentialSTDP(C, taup=100*ms, taum=100*ms, Ap=0.1, Am=0.1, wmax=10)

M = MultiStateMonitor(stp.vars, record=True)
run(1*second)

subplot(211)
M.plot()
legend()
subplot(212)
M2.plot()
show()