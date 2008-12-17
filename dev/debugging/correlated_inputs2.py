'''
An example with correlated spike trains
From: Brette, R. (2007). Generation of correlated spike trains.
'''
from brian import *
from brian.correlatedspikes import *
from brian.utils.statistics import *

N=2
r0=15*Hz+20*rand(N)*Hz # rates
C=50*rand(N,N)*Hz**2 # correlation matrix
C=C+C.T
input=CorrelatedSpikeTrains(N,rates=r0,C=C,tauc=10*ms)

S=SpikeMonitor(input)
duration=200*second
run(duration)

subplot(211)
plot(CCVF(S[0],S[1],width=20*ms,bin=1*ms,T=duration))
subplot(212)
plot(CCF(S[0],S[1],width=20*ms,bin=1*ms,T=duration))
print C[0,1],r0[0],r0[1]
show()
