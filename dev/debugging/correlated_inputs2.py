'''
An example with correlated spike trains
From: Brette, R. (2008). Generation of correlated spike trains. Neural Computation.
'''
from brian import *
from brian.correlatedspikes import *
from brian.utils.statistics import *

N=2
r0=25*Hz+30*rand(N)*Hz # rates
C=500*rand(N,N)*Hz**2 # correlation matrix
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
print total_correlation(S[0],S[1],T=duration),2*.01*C[0,1]/r0[0]
print CV(S[0])
show()
