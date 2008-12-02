# coding: latin-1
"""
This is a Brian script implementing a benchmark described
in the following review paper:

Simulation of networks of spiking neurons: A review of tools and strategies (2007).
Brette, Rudolph, Carnevale, Hines, Beeman, Bower, Diesmann, Goodman, Harris, Zirpe,
Natschläger, Pecevski, Ermentrout, Djurfeldt, Lansner, Rochel, Vibert, Alvarez, Muller,
Davison, El Boustani and Destexhe.
Journal of Computational Neuroscience 23(3):349-98

Benchmark 1: random network of integrate-and-fire neurons with exponential synaptic conductances

Clock-driven implementation with Euler integration
(no spike time interpolation)

R. Brette - Dec 2007
--------------------------------------------------------------------------------------
Brian is a simulator for spiking neural networks written in Python, developed by
R. Brette and D. Goodman.
http://brian.di.ens.fr
"""

from brian import *
from brian.experimental.ccodegen import AutoCompiledNonlinearStateUpdater
import time

duration = 1*second
N = 32000
use_new_nonlinear = False
useconn = False

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

P=NeuronGroup(N,model=eqs,threshold=10*mvolt,\
              reset=0*mvolt,refractory=5*msecond,
              order=1,compile=True)
if use_new_nonlinear:
    P._state_updater = AutoCompiledNonlinearStateUpdater(P._state_updater.eqs, clock=P.clock, freeze=True)
Pe=P.subgroup(int(N*0.8))
Pi=P.subgroup(N-len(Pe))
if useconn:
    Ce=Connection(Pe,P,'ge')
    Ci=Connection(Pi,P,'gi')
    we=6./10. # excitatory synaptic weight (voltage)
    wi=67./10. # inhibitory synaptic weight
    Ce.connect_random(Pe, P, 80./N, weight=we)
    Ci.connect_random(Pi, P, 80./N, weight=wi)
# Initialization
P.v=(randn(len(P))*5-5)*mvolt
P.ge=randn(len(P))*1.5+4
P.gi=randn(len(P))*12+20

# Record the number of spikes
if useconn:
    Me=PopulationSpikeCounter(Pe)
    Mi=PopulationSpikeCounter(Pi)

start_time=time.time()

run(duration)
simtime=time.time()-start_time
print "N =", N
print "Duration", duration
print "Using new C++ nonlinear state updater", use_new_nonlinear
print "Using connections", useconn
print "Simulation time:",simtime,"seconds"
if useconn:
    print Me.nspikes,"excitatory spikes"
    print Mi.nspikes,"inhibitory spikes"
