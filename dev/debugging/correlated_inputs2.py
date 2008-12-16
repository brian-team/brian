'''
An example with correlated spike trains
From: Brette, R. (2007). Generation of correlated spike trains.
'''
from brian import *
from brian.correlatedspikes import *
from brian.utils.statistics import *

N=10
r0=15*Hz+20*rand(N)*Hz # rates
C=1.5*(5+rand(N,N))*Hz**2 # correlation matrix
C=C+C.T
input=CorrelatedSpikeTrains(N,rates=r0,C=C,tauc=10*ms)

S=SpikeMonitor(input)
duration=100*second
run(duration)

subplot(211)
raster_plot(S)
subplot(212)
plot(CCVF(S[0],S[1],width=20*ms))
print C[0,1]
show()
