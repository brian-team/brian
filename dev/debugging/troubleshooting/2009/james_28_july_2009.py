from brian import *
from brian.library.synapses import *

taue = 5 * ms
gl = 20 * nS
Ee = 0 * mV
El = -60 * mV
Vt = -50 * mV
Vr = -60 * mV

eqs = MembraneEquation(C=200 * pF)
eqs += Current('I=gl*(El-vm):amp')
eqs += alpha_conductance(input='ge', E=Ee, tau=taue)

input = PoissonGroup(2, rates=50 * Hz)
G = NeuronGroup(N=1, model=eqs, threshold=Vt, reset=Vr)

C = Connection(input, G, 'ge')
C[0, 0] = 10 * nS
C[1, 0] = 5 * nS

Mvm = StateMonitor(G, 'vm', record=True)
Mge = StateMonitor(G, 'ge_current', record=True)

run(100 * ms)

figure()
subplot(211)
plot(Mvm.times, Mvm[0])
subplot(212)
plot(Mge.times, Mge[0])
show()
