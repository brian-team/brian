#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Brette R (2012). Computing with neural synchrony. PLoS Comp Biol. 8(6): e1002561. doi:10.1371/journal.pcbi.1002561
------------------------------------------------------------------------------------------------------------------
Figure 5E. (very long simulation)

Caption (Fig 5E). Precision and reliability of spike timing as a function of SNR.

Simulations are run in parallel on all cores but one.
"""
from brian import *
import multiprocessing

def autocor(PSTH,N=None,T=20*ms,bin=None):
    '''
    Autocorrelogram of PSTH, to calculate a shuffled autocorrelogram
    
    N = number of spike trains
    T = temporal window
    bin = PSTH bin
    The baseline is not subtracted.
    
    Returns times,SAC
    '''
    if bin is None:
        bin=defaultclock.dt
    n=len(PSTH)
    p=int(T/bin)
    SAC=zeros(p)
    if N is None:
        SAC[0]=mean(PSTH*PSTH)
    else: # correction to exclude self-coincidences
        PSTHnoself=clip(PSTH-1./(bin*N),0,Inf)
        SAC[0]=mean(PSTH*PSTHnoself)*N/(N-1.)
    SAC[1:]=[mean(PSTH[:-i]*PSTH[i:]) for i in range(1,p)]
    SAC=hstack((SAC[::-1],SAC[1:]))
    return (arange(len(SAC))-len(SAC)/2)*bin,SAC

def halfwidth(x):
    '''
    Returns half-width of function given by x in bin numbers.
    This is used to calculate the precision (left panel).
    '''
    M,n=max(x),argmax(x)
    return find(x[n:]<M/2)[0]+n-find(x[:n]<M/2)[-1]

def reproducibility(SNR):
    '''
    Calculates the precision (timescale) and reliability (strength) for a given
    signal-to-noise ratio.
    '''
    sys.stdout.write("SNR:"+str(SNR)+'\n')
    sys.stdout.flush() # we use this instead of print because of multiprocessing
    reinit_default_clock() # important because we do multiple simulations
    # The common noisy input
    N=5000                 # number of neurons simultaneously simulated
    duration=30*second     # duration of one simulation, 200 seconds in the paper
    tau_noise=5*ms
    input=NeuronGroup(1,model='dx/dt=-x/tau_noise+(2./tau_noise)**.5*xi:1')
    
    # The noisy neurons receiving the same input
    tau=10*ms
    sigma=.5 # input amplitude
    Z=sigma*sqrt((tau_noise+tau)/(tau_noise*(SNR**2+1))) # normalizing factor
    eqs_neurons='''
    dx/dt=(Z*(SNR*I+u)-x)/tau:1
    du/dt=-u/tau_noise+(2./tau_noise)**.5*xi:1
    I : 1
    '''
    neurons=NeuronGroup(N,model=eqs_neurons,threshold=1,reset=0,refractory=5*ms)
    neurons.x=rand(N) # random initial conditions
    neurons.I=linked_var(input,'x')
    rate=PopulationRateMonitor(neurons) # PSTH
    
    run(duration)
    
    t,SAC=autocor(rate.rate,N,T=30*ms)
    timescale=float(halfwidth(SAC-mean(rate.rate)**2))*defaultclock.dt # precision
    strength=sum(SAC-mean(rate.rate)**2)*float(defaultclock.dt)/mean(rate.rate) # reliability
    
    return timescale,strength

if __name__=='__main__':
    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1) # all cores but one
    SNRdB= linspace(-10,15,20) # 100 points in the paper
    SNR = 10.**(SNRdB/10.)
    results = pool.map(reproducibility, SNR) # launches multiple processes
    timescale,strength=zip(*results)

    # Figure
    subplot(211)
    plot(SNRdB,timescale*1000)
    xlabel('SNR (dB)')
    ylabel('Precision (ms)')
    subplot(212)
    plot(SNRdB,strength*100)
    xlabel('SNR (dB)')
    ylabel('Reliability (%)')
    show()
