#!/usr/bin/env python
"""
Analysis of spike threshold.

Loads a current clamp voltage trace, compensates (remove electrode voltage)
and analyses the spikes.
"""
from brian import *
from brian.library.electrophysiology import *
import numpy

dt=.1*ms
Vraw = numpy.load("trace.npy") # Raw current clamp trace
I = numpy.load("current.npy")
V, _ = Lp_compensate(I, Vraw, dt) # Electrode compensation

# Peaks
spike_criterion=find_spike_criterion(V)
print "Spike detected when V exceeds",float(spike_criterion/mV),"mV"
peaks=spike_peaks(V,vc=spike_criterion) # vc is optional

# Onsets (= spike threshold)
onsets=spike_onsets(V,criterion=3*dt,vc=spike_criterion) # Criterion: dV/dt>3 V/s

# Spike-triggered average of V
STA=spike_shape(V, onsets=onsets, before=100, after=100)

print "Spike duration:",float(spike_duration(V,onsets=onsets)*dt/ms),"ms"
print "Reset potential:",float(reset_potential(V,peaks=peaks)/mV),"mV"

# Spike threshold statistics
slope=slope_threshold(V,onsets=onsets,T=int(5*ms/dt))

# Subthreshold trace
subthreshold=-spike_mask(V)

t=arange(len(V))*dt
subplot(221)
plot(t/ms,V/mV,'k')
plot(t[peaks]/ms,V[peaks]/mV,".b")
plot(t[onsets]/ms,V[onsets]/mV,".r")
subplot(222)
plot(((arange(len(STA))-100)*dt)/ms,STA/mV,'k')
subplot(223)
plot(t[subthreshold]/ms,V[subthreshold]/mV,'k')
subplot(224)
plot(slope/ms,V[onsets]/mV,'.')
show()
