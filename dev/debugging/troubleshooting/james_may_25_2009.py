from brian import *
from brian.library.IF import *

inputGroup = PoissonGroup(500, 54)
Mall       = SpikeMonitor(inputGroup)
run(5000*ms)

# XXX

eqs = Izhikevich(a=0.02/ms, b=0.2/ms)
Ginput  = SpikeGeneratorGroup(500, Mall.spikes)
Goutput = NeuronGroup(1, model=eqs, threshold=30*mV,
            reset=AdaptiveReset(Vr=-65*mV, b=0.0805*nA))

Mvolt = StateMonitor(Goutput,'vm',record=0)

C = Connection(Ginput, Goutput, state='vm')
C.connect_full(weight=0.475*mV)
Ginput.v  = -65*mV
Goutput.v = -65*mV

stdp = ExponentialSTDP(C, 33*ms, 16.8*ms, 0.01, -0.01, wmax=0.01)

reinit_default_clock()
run(5000*ms)

plot(Mvolt.times/(1*msecond),Mvolt[0]/(1*mvolt))
show()