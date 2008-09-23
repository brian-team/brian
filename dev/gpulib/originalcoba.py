# coding: latin-1
"""
This is an implementation of a benchmark described
in the following review paper:

Simulation of networks of spiking neurons: A review of tools and strategies (2006).
Brette, Rudolph, Carnevale, Hines, Beeman, Bower, Diesmann, Goodman, Harris, Zirpe,
Natschläger, Pecevski, Ermentrout, Djurfeldt, Lansner, Rochel, Vibert, Alvarez, Muller,
Davison, El Boustani and Destexhe.
Journal of Computational Neuroscience

Benchmark 1: random network of integrate-and-fire neurons with exponential synaptic conductances

Clock-driven implementation with Euler integration
(no spike time interpolation)

R. Brette - Dec 2007

About 5.5s.
With compilation: 4s
"""

#import brian_no_units

from brian import *
import time
#set_global_preferences(useweave=True)

# Time constants
taum=20*msecond
taue=5*msecond
taui=10*msecond
# Reversal potentials
Ee=(0.+60.)*mvolt
Ei=(-80.+60.)*mvolt

start_time=time.time()
eqs=Equations('''
dv/dt = (-v+ge*(Ee-v)+gi*(Ei-v))*(1./taum) : volt
dge/dt = -ge*(1./taue) : 1
dgi/dt = -gi*(1./taui) : 1 
''')
# NB 1: conductances are in units of the leak conductance
# NB 2: multiplication is faster than division

P=NeuronGroup(4000,model=eqs,threshold=10*mvolt,\
              reset=0*mvolt,refractory=5*msecond,
              order=1,compile=True)
#STOPPPPPP
Pe=P.subgroup(3200)
Pi=P.subgroup(800)
Ce=Connection(Pe,P,'ge')
Ci=Connection(Pi,P,'gi')
we=6./10. # excitatory synaptic weight (voltage)
wi=67./10. # inhibitory synaptic weight
Ce.connect_random(Pe, P, 0.02,weight=we)
Ci.connect_random(Pi, P, 0.02,weight=wi)
# Initialization
P.v=(randn(len(P))*5-5)*mvolt
P.ge=randn(len(P))*1.5+4
P.gi=randn(len(P))*12+20

# Record the number of spikes
Me=PopulationSpikeCounter(Pe)
Mi=PopulationSpikeCounter(Pi)

print "Network construction time:",time.time()-start_time,"seconds"
print "Simulation running..."
start_time=time.time()

run(1*second)
duration=time.time()-start_time
print "Simulation time:",duration,"seconds"
print Me.nspikes,"excitatory spikes"
print Mi.nspikes,"inhibitory spikes"
