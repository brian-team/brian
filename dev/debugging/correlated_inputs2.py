'''
An example with correlated spike trains
From: Brette, R. (2007). Generation of correlated spike trains.
'''
from brian import *
from brian.correlatedspikes import *

def crosscorrelogram(T1,T2,tmax=20*ms):
    '''
    Cross-correlogram between two spike trains T1 and T2.
    Very slow!
    '''
    X1=dot(T1.reshape((len(T1),1)),ones((1,len(T2))))
    X2=dot(ones((len(T1),1)),T2.reshape((1,len(T2))))
    M=(X1-X2)
    ind=(M<tmax) & (M>-tmax)
    return M[ind].flatten()

N=10
r0=15*Hz+20*rand(N)*Hz # rates
C=1.5*(5+rand(N,N))*Hz**2 # correlation matrix
C=C+C.T
input=CorrelatedSpikeTrains(N,rates=r0,C=C,tauc=10*ms)

S=SpikeMonitor(input)
counter=SpikeCounter(input)
S2=PopulationRateMonitor(input)
M=StateMonitor(input,'rate',record=0)
duration=1000*ms
run(duration)
subplot(211)
raster_plot(S)
subplot(212)
plot(S2.times/ms,S2.smooth_rate(5*ms))
plot(M.times/ms,M[0]/Hz)
show()
