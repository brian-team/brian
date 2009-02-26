'''
An auditory filterbank implemented with Poisson neurons
'''
from brian import *

defaultclock.dt=.01*ms

N=500
tau=1*ms # Decay time constant of filters = 2*tau
freq=linspace(100*Hz,2000*Hz,N) # characteristic frequencies
f_stimulus=1000*Hz # stimulus frequency
gain=500*Hz

eqs='''
dv/dt=(-a*w-v+I)/tau : Hz
dw/dt=(v-w)/tau : Hz # e.g. linearized potassium channel with conductance a
a : 1
I = gain*sin(2*pi*f_stimulus*t) : Hz
'''

neurones=NeuronGroup(N,model=eqs,threshold=PoissonThreshold())
neurones.a=(2*pi*freq*tau)**2

spikes=SpikeMonitor(neurones)
counter=SpikeCounter(neurones)
run(300*ms)

subplot(211)
raster_plot(spikes)
subplot(212)
plot(freq,counter.count)
show()
