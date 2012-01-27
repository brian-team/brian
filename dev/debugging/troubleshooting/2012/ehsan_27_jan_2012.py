from brian import *
from brian.experimental.synapses import *
Vth, Vr, El = 0 * mV, -60*mV, -70*mV

eqs = Equations('''
dVm/dt = (El - Vm + Pse*(Vm-0*mV) - Psi*(Vm+80*mV))/(20*msecond) :volt
Pse :1
Psi :1
''')

Excit = NeuronGroup(1, eqs, threshold = Vth, reset=Vr)
Inhib = NeuronGroup(1, eqs, threshold = Vth, reset=Vr)

SynapsE2I=Synapses(Excit, Inhib, model='''dPse/dt = -Pse/(5*msecond) : 1
w : 1    ''')
Inhib.Pse=SynapsE2I.Pse
