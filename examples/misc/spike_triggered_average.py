#!/usr/bin/env python
'''
Example of the use of the function spike_triggered_average. A white noise  is filtered
by a gaussian filter (low pass filter) which output is used to generate spikes (poission process)
Those spikes are used in conjunction with the input signal to retrieve the filter function.
'''
from brian import *
from brian.hears import *
from numpy.random import randn
from numpy.linalg import norm
from matplotlib import pyplot

dt = 0.1*ms
defaultclock.dt = dt
stimulus_duration = 15000*ms
stimulus = randn(int(stimulus_duration/     dt))

#filter
n=200
filt = exp(-((linspace(0.5,n,n))-(n+5)/2)**2/(n/3));
filt = filt/norm(filt)*1000;
filtered_stimulus = convolve(stimulus,filt)


neuron = PoissonGroup(1,lambda t:filtered_stimulus[int(t/dt)])

spikes = SpikeMonitor(neuron)
run(stimulus_duration,report='text')
spikes = spikes[0] #resulting spikes

max_interval = 20*ms #window duration of the spike triggered average
onset = 10*ms
sta,time_axis = spike_triggered_average(spikes,stimulus,max_interval,dt,onset=onset,display=True)


figure()
plot(time_axis,filt/max(filt))
plot(time_axis,sta/max(sta))
xlabel('time axis')
ylabel('sta')
legend(('real filter','estimated filter'))

show()
