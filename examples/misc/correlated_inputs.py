'''
An example with correlated spike trains
From: Brette, R. (2007). Generation of correlated spike trains.
'''
from brian import *
from brian.correlatedspikes import *

input=HomogeneousCorrelatedSpikeTrains(1000, r=10*Hz, c=0.1, tauc=10*ms)

S=SpikeMonitor(input)
S2=PopulationRateMonitor(input)
M=StateMonitor(input, 'rate', record=0)
run(1000*ms)
subplot(211)
raster_plot(S)
subplot(212)
plot(S2.times/ms, S2.smooth_rate(5*ms))
plot(M.times/ms, M[0]/Hz)
show()
