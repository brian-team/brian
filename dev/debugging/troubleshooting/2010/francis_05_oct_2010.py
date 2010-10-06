# Neural Cortical Model of a single neuron with poisson input
from brian import *
from brian.library.synapses import *
import time
import random

defaultclock.dt=0.01*ms

vr    = -90*mvolt   # reset/resting membrane potential
vt    = -55*mvolt   # threshold
ref   =  1*msecond  # refractory period

tauge = 0.25*msecond # excitatory conductance time constant

El    = -70*mvolt   # resting potential
Ee    =   0*mvolt   # reversal potential of excitatory ionic channels

gl    = 10.*nsiemens # leak conductance

#compute the somatic and dendritic capacitance for each neuron type
Rmse = 600*Mohm
taue  =  10*msecond # somatic excitatory membrane time constant
Ces = taue/Rmse # excitatory cell somatic capacitance

#Poisson input value
Pinput = 150.5*nS # Highly exaggerated!
Pfreq  = 50*kHz

# Number of neurons
N   = 1

###### Neuron model
exc_soma = MembraneEquation(C=Ces) +  Current('I=gl*(El-vm):amp')
exc_soma += alpha_conductance(input='ge' ,E=Ee, tau=tauge)

print exc_soma

# Create Excitatory Neural Sheet
NG  = NeuronGroup(N, model=exc_soma, threshold='vm>vt',reset=vr,refractory=ref)

###### Make the Connections
#Poisson Input
PG  = PoissonGroup(N, N*[Pfreq])

P = Connection(PG, NG, 'ge')
P.connect_one_to_one(PG, NG, weight=Pinput)

SpP = SpikeMonitor(PG)
SpNG = SpikeMonitor(NG)

M=StateMonitor(NG,'vm',record=True)

#status
print 'Running...'
run(1*second)

###### analysis output

print SpP.nspikes, '\tpoisson spikes'
print SpNG.nspikes, '\ttotal spikes'

subplot(221)
raster_plot(SpP)
ylabel('Poisson Neuron Id')

subplot(223)
raster_plot(SpNG)
ylabel('Neuron Id')

subplot(222)
plot(M.times/ms,M[0]/mV)

show()
