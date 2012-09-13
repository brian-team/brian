#!/usr/bin/env python
'''
Example of how to use the van Rossum metric. 

The VanRossumMetric function, which is defined as a monitor and therefore works online, 
computes  the metric between every neuron in a given population. The present example show 
the concept of phase locking:  N neurons  are driven by  sinusoidal inputs with different amplitude.

 Use: output=VanRossumMetric(source, tau=4 * ms)
 
 source is a NeuronGroup of N neurons
 tau is the time constant of the kernel used in the metric
 
 output is a monitor with attribute distance which is the distance matrix between the neurons in source
'''
from brian import *
from time import time

tau=20*ms
N=100
b=1.2 # constant current mean, the modulation varies
f=10*Hz
delta =2*ms

eqs='''
dv/dt=(-v+a*sin(2*pi*f*t)+b)/tau : 1
a : 1
'''

neurons=NeuronGroup(N,model=eqs,threshold=1,reset=0)
neurons.v=rand(N)
neurons.a=linspace(.05,0.75,N)
S=SpikeMonitor(neurons)
trace=StateMonitor(neurons,'v',record=50)

van_rossum_metric=VanRossumMetric(neurons, tau=4 * ms)

run(1000*ms)

raster_plot(S)
title('Raster plot')

figure()
title('Distance matrix between spike trains')
imshow(van_rossum_metric.distance)
colorbar()
show()