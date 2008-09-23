# Test of membrane equations
from brian import *
from brian.library.IF import *

myclock=Clock(dt=.1*ms)

eqs=Izhikevich()
eqs+=InjectedCurrent('I:volt/second')

neuron=NeuronGroup(1,model=eqs,threshold=30*mV,reset=AdaptiveReset(Vr=-65*mV,b=2*mV/ms))
neuron.vm=-65*mV
neuron.I=20*mV/ms
mon=StateMonitor(neuron,'vm',record=True)

run(500*ms)
plot(mon.times/ms,mon[0]/mV)
show()
