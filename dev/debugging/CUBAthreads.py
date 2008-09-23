# coding: latin-1
"""
This is an implementation of a benchmark described
in the following review paper:

Simulation of networks of spiking neurons: A review of tools and strategies (2006).
Brette, Rudolph, Carnevale, Hines, Beeman, Bower, Diesmann, Goodman, Harris, Zirpe,
Natschläger, Pecevski, Ermentrout, Djurfeldt, Lansner, Rochel, Vibert, Alvarez, Muller,
Davison, El Boustani and Destexhe.
Journal of Computational Neuroscience

Benchmark 2: random network of integrate-and-fire neurons with exponential synaptic currents

Clock-driven implementation with exact subthreshold integration
(but spike times are aligned to the grid)

R. Brette - Oct 2007
"""

from brianUnitPrefs import turn_off_units
turn_off_units()

from brian import *
import time
from brian.stdunits import *
from scipy import rand

import c_profile

def main():
    clk=Clock(t=0*ms,dt=0.1*ms)
    
    start_time=time.time()
    taum=20*ms
    taue=5*ms
    taui=10*ms
    Vt=-50*mV
    Vr=-60*mV
    El=-49*mV
    dv=lambda v,ge,gi: (ge+gi-(v-El))/taum
    dge=lambda v,ge,gi: -ge/taue
    dgi=lambda v,ge,gi: -gi/taui
    
    P1=NeuronGroup(2000,model=(dv,dge,dgi),threshold=Vt,reset=Vr,\
                  refractory=5*ms,clock=clk)
    P2=NeuronGroup(2000,model=(dv,dge,dgi),threshold=Vt,reset=Vr,\
                  refractory=5*ms,clock=clk)
    P2e=P2.subgroup(1200)
    P2i=P2.subgroup(800)
    Ce1=Connection(P1,P1,1)
    Ce2=Connection(P1,P2,1)
    Ce3=Connection(P2e,P1,1)
    Ce4=Connection(P2e,P2,1)
    Ci1=Connection(P2i,P1,2)
    Ci2=Connection(P2i,P2,2)
    we=(60*0.27/10)*mV # excitatory synaptic weight (voltage)
    wi=(-20*4.5/10)*mV # inhibitory synaptic weight
    Ce1.connect_random(P1, P1, 0.02,weight=we)
    Ce2.connect_random(P1, P2, 0.02,weight=we)
    Ce3.connect_random(P2e, P1, 0.02,weight=we)
    Ce4.connect_random(P2e, P2, 0.02,weight=we)
    Ci1.connect_random(P2i, P1, 0.02,weight=wi)
    Ci2.connect_random(P2i, P2, 0.02,weight=wi)
    P1.Vm=Vr+rand(len(P1))*(Vt-Vr)
    P2.Vm=Vr+rand(len(P2))*(Vt-Vr)
    
    # Record the number of spikes
    M1=SpikeMonitor(P1)
    M2=SpikeMonitor(P2)
    
    net=Network(groups=[P1,P2],connections=[Ce1,Ce2,Ce3,Ce4,Ci1,Ci2,M1,M2])
    
    print "Network construction time:",time.time()-start_time,"seconds"
    print len(net),"neurons in the network"
    print "Simulation running..."
    start_time=time.time()
    
    net.run(1*second,threads=2)
    
    print "Simulation time:",time.time()-start_time,"seconds"
    print M1.nspikes,"spikes in P1"
    print M2.nspikes,"spikes in P2"

#c_profile.run('main()')
main()
