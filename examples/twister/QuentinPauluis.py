#!/usr/bin/env python
'''
Quentin Pauluis's entry for the 2012 Brian twister.
'''
from brian import *

taum = 20 * ms          # membrane time constant
taue = 5 * ms          # excitatory synaptic time constant
taui = 10 * ms          # inhibitory synaptic time constant
Vt = -50 * mV          # spike threshold
Vr = -60 * mV          # reset value
El = -49 * mV          # resting potential
we = (60 * 0.27 / 10) * mV # excitatory synaptic weight
wi = (20 * 4.5 / 10) * mV # inhibitory synaptic weight
eqs = Equations('''
        dV/dt  = (ge-gi-(V-El))/taum : volt
        dge/dt = -ge/taue            : volt
        dgi/dt = -gi/taui            : volt
        ''')
G = NeuronGroup(4000, model=eqs, threshold=Vt, reset=Vr)
Ge = G.subgroup(3200) # Excitatory neurons
Gi = G.subgroup(800)  # Inhibitory neurons
Ce = Connection(Ge, G, 'ge', sparseness=0.2, weight=we)

Ci = Connection(Gi, G, 'gi', sparseness=0.2, weight=wi*10)
Cii=Connection(Gi,Gi,'gi',sparseness=0.2, weight=wi)
M = SpikeMonitor(G)
E=SpikeMonitor(Ge,'+')
I=SpikeMonitor(Gi,'o')
MV = StateMonitor(G, 'V', record=0)
Mge = StateMonitor(G, 'ge', record=0)
Mgi = StateMonitor(G, 'gi', record=0)
G.V = Vr + (Vt - Vr) * rand(len(G))
run(2500 * ms)
subplot(211)
raster_plot(M, title='The CUBA network', newfigure=False)
raster_plot(E)
raster_plot(I)
subplot(223)
plot(MV.times / ms, MV[0] / mV)
xlabel('Time (ms)')
ylabel('V (mV)')
subplot(224)
plot(Mge.times / ms, Mge[0] / mV)
plot(Mgi.times / ms, Mgi[0] / mV)
xlabel('Time (ms)')
ylabel('ge and gi (mV)')
legend(('ge', 'gi'), 'upper right')
show()
#new.Figure













