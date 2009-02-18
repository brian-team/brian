'''
Adapted (and simplified) from
Theory of Arachnid Prey Localization
W. Sturzl, R. Kempter, and J. L. van Hemmen
PRL 2000
'''
from brian import *

# Parameters
degree=2*pi/360.
R=2.5*cm # radius of scorpion
vr=50*meter/second # Rayleigh wave speed
phi=34*degree # angle of prey
deltaI=.7*ms # inhibitory delay
gamma=(22.5+45*arange(8))*degree # leg angle

wave=lambda t:.2*sin(2*pi*300*Hz*t)*cos(2*pi*25*Hz*t)

# Leg mechanical receptors
tau_legs=1*ms
sigma=.01
eqs_legs="""
dv/dt=(1+wave(t-d)-v)/tau_legs+sigma*(2./tau_legs)**.5*xi:1
d : second
"""
legs=NeuronGroup(8,model=eqs_legs,threshold=1,reset=0,refractory=1*ms)
legs.d=R/vr*(1-cos(phi-gamma))  # wave delay
spikes_legs=SpikeCounter(legs)

# Command neurons
tau=1*ms
taus=1*ms
wex=7
winh=-2
eqs_neuron='''
dv/dt=(x-v)/tau : 1
dx/dt=(y-x)/taus : 1 # alpha currents
dy/dt=-y/taus : 1
'''
neurons=NeuronGroup(8,model=eqs_neuron,threshold=1,reset=0)
synapses_ex=IdentityConnection(legs,neurons,'y',weight=wex)
synapses_inh=Connection(legs,neurons,'y',delay=deltaI)
for i in range(8):
    synapses_inh[i,(4+i-1)%8]=winh
    synapses_inh[i,(4+i)%8]=winh
    synapses_inh[i,(4+i+1)%8]=winh
spikes=SpikeCounter(neurons)

run(200*ms)
nspikes=spikes.count
x=sum(nspikes*exp(gamma*1j))
print "Angle (deg):",arctan(imag(x)/real(x))/degree
#polar(concatenate((gamma,[gamma[0]+2*pi])),concatenate((nspikes,[nspikes[0]]))/(200*ms))
polar(gamma,nspikes)
show()
