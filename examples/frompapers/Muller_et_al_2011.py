#!/usr/bin/env python
"""
Interplay of STDP and input oscillations
----------------------------------------
Figure 4 from:
Muller L, Brette R and Gutkin B (2011) Spike-timing dependent plasticity and 
feed-forward input oscillations produce precise and invariant spike phase-locking. Front. 
Comput. Neurosci. 5:45. doi: 10.3389/fncom.2011.00045

Description:
In this simulation, a group of IF neurons is given a tonic DC input and a tonic AC input.
The DC input is mediated by current injection (neurons.I, line 62), and the AC input is 
mediated by Poisson processes whose rate parameters are oscillating in time. Each neuron in 
the group is given a different DC input, ensuring a unique initial phase. After two seconds 
of simulation (to integrate out any initial transients), the STDP rule is turned on 
(ExponentialSTDP, line 68), and the population of neurons converges to the theoretically 
predicted fixed point. As there is some noise in the phase due to the random inputs, the 
simulation is averaged over trials (50 in Figure 4, though 10 trials should be fine for 
testing).

The trials run in parallel on all available processors (10 trials take about
2 minutes on a modern PC).
"""

### IMPORTS
from brian import *
import multiprocessing

### PARAMETERS
N=5000
M=10
taum=33*ms                          
tau_pre=20*ms
tau_post=tau_pre
Ee=0*mV
vt=-54*mV
vr=-70*mV
El=-70*mV
taue=5*ms
f=20*Hz
theta_period = 1/f
Rm=200*Mohm
a = linspace(51,65,num=M)       
weights = .001
ratio=1.50
dA_pre=.01
dA_post=.01*ratio 
trials=10

### SIMULATION LOOP
def trial(n): # n is the trial number
    reinit_default_clock()
    clear(True)

    eqs_neurons='''
    dv/dt=((ge*(Ee-vr))+Rm*I+(El-v))/taum : volt   
    dge/dt=-ge/taue : 1
    I : amp
    '''

    inputs = PoissonGroup(N,rates=lambda t:((.5-.5*cos(2*pi*f*t)))*10*Hz)           
    neurons=NeuronGroup(M,model=eqs_neurons,threshold=vt,reset=vr)
    neurons.I = a*pA
    synapses=Connection(inputs,neurons,'ge',weight=weights)
    neurons.v=vr

    S = SpikeMonitor(neurons)
    run(2*second)
    stdp=ExponentialSTDP(synapses,tau_pre,tau_post,dA_pre,-dA_post,wmax=10*weights,interactions='all',update='additive')     
    run(5*second)
    
    phase=zeros((M,200))
    for b in range(0,M):    
        tmp_phase=(S[b]%theta_period)*(360/theta_period)
        phase[b,range(0,len(tmp_phase))] = tmp_phase
        
    return phase

if __name__=='__main__': # This is very important on Windows, otherwise the machine crashes!
    phase = zeros((M,200,trials))
    
    print "This will take approximately 2 minutes."
    pool=multiprocessing.Pool() # uses all available processors b
    results=pool.map(trial,range(trials))
    for i in range(trials):
        phase[:,:,i]=results[i]
    
    ### PLOTTING
    for b in range(0,M):
        m = mean(phase[b,:,:],axis=1)
        st = std(phase[b,:,:],axis=1)/sqrt(trials)
        errorbar(range(0,135), m[range(0,135)], yerr=st[range(0,135)], xerr=None,
             fmt='-', ecolor=None, elinewidth=None, capsize=3,
             barsabove=False, lolims=False, uplims=False,
             xlolims=False, xuplims=False)
    
    title('STDP + Oscillations Simulation')
    xlabel('Spike Number')
    ylabel('Spike Phase (deg)')
    xlim([0, 135])
    ylim([140, 280])
    
    show()