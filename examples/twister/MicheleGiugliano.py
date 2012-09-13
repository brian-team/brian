#!/usr/bin/env python
'''
Michele Giugliano's entry for the 2012 Brian twister.
'''
#
# Figure5B - from Giugliano et al., 2004
# Journal of Neurophysiology 92(2):977-96
#
# implemented by Eleni Vasilaki <e.vasilaki@sheffield.ac.uk> and
# Michele Giugliano <michele.giugliano@ua.ac.be>
#
# A sparsely connected network of excitatory neurons, interacting
# via current-based synaptic interactions, and incorporating 
# spike-frequency adaptation, is simulated.
#
# Its overall emerging firing rate activity replicates some of the features of
# spontaneous patterned electrical activity, observed experimentally in cultured 
# networks of neurons dissociated from the neocortex.
#
from brian import *

# Parameters of the simulation
T    = 30000 *ms     # life time of the simulation
N    = 100           # total number of (excitatory) integrate-and-fire model neurons in the network

# Parameters of each model neuron, voltage dynamics
C    = 67.95 *pF      # Membrane capacitance of single model neurons
tau  = 22.25  *ms     # Membrane time-constant of single model neurons
H    = 2.39 *mV       # Reset voltage, mimicking hyperpolarization potential following a spike
theta= 20 *mV         # Threshold voltage for spike initiation
tauarp=7.76 *ms       # Absolute refractory period

# Parameters of each model neuron, spike-frequency adaptation dynamics
taua = 2100 *ms       # Adaptation time constant
a    = 0.75 *pA       # Adaptation scaling factor - NO ADAPTATION
D    = 1*ms           # Unit consistency factor        
temp = 1. *ms**(-.5)  # Unit consistency factor         

# Parameters of network connectivity
Cee  = 0.38           # Sparseness of all-to-all random connectivity
taue = 5 *ms          # Decay time constant of excitatory EPSPs
delta= 1.5 * ms       # Conduction+synaptic propagation delay
J    = 14.5* pA       # Strenght of synaptic coupling, up to 18 *pA

# Parameters of background synaptic activity, modelled as a identical and independent noisy extra-input to each model neuron
m0   = 25.1 *pA       # Mean background input current
s0   = 92 *pA         # Std dev of the (noisy) background input current

# Each model neuron is described as a leaky integrate-and-fire with adaptation and current-driven synapses
eqs = """
dv/dt  = - v / tau - a/C * x  + Ie/C + (m0 + s0 * xi / temp)/C  : mV
dx/dt  = -x/taua   : 1
dIe/dt = -Ie/taue  : pA
"""

# Custom refractory mechanisms are employed here, to allow the membrane potential to be clamped to the reset value H
def myresetfunc(P, spikes):
 P.v[spikes] = H   #reset voltage 
 P.x[spikes] += 1  #low pass filter of spikes (adaptation mechanism)

SCR = SimpleCustomRefractoriness(myresetfunc, tauarp, state='v')
 
# The population of identical N model neuon is defined now
P = NeuronGroup(N, model=eqs, threshold=theta, reset=SCR)

# The interneuronal connectivity is defined now
Ce = Connection(P, P, 'Ie', weight=J, sparseness=Cee, delay=delta)

# Initialization of the state variables, for each model neuron
P.v    = rand(len(P)) * 20 * mV  #membrane potential
P.x    = rand(len(P)) * 2        #low pass filter of spikes
P.Ie   = 0 *pA                   #excitatory synaptic input

# Definition of tools for plotting and visualization of single neuron and population quantities
R      = PopulationRateMonitor(P)
M      = SpikeMonitor(P)
trace  = StateMonitor(P, 'v', record=0)
tracex = StateMonitor(P, 'x', record=0)

print "Simulation running... (long-lasting simulation: be patient)"
run(T)

print "Simulation completed! If you did not see any firing rate population burst (lower panel), then slightly increase J!"

# Plot nice spikes - adapted from Brette's code
vm       = trace[0]
spikes0  = [t for i,t in M.spikes if i==0]
for i in range(0,len(spikes0)):
    k = int(spikes0[i] / defaultclock.dt)
    vm[k] = 80 * mV
    
subplot(311) #membrane potential of neuron 0
plot(trace.times / ms, vm / mV - 60)

subplot(312) #raster plot
raster_plot(M) 

subplot(313) #smoothed population rate 
plot(R.times / ms, R.smooth_rate(5*ms) / Hz, tracex.times / ms, tracex[0] * 10)
ylim(0, 120)

show()

