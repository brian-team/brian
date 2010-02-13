'''
Hansel et al (1998)
On Numerical Simulations of Integrate-and-Fire Neural Networks.
Neural Computation

The authors show that a very small time step is necessary to
obtain some synchronization results.
The model is a network of coupled oscillators (all-to-all).
'''
from brian import *

#defaultclock.dt=.01*ms
N=128
duration1=2*second
duration2=2*second
Vl=-60*mV
theta=-40*mV
tau=10*ms
tau1=3*ms
tau2=1*ms
I0=21*mV # I couldn't find the value in the text!
# there is also an error in the text: replace tau*I0 by I0
w=.7*mV # slightly below critical value
eqs='''
dV/dt=(Vl-V+I+I0)/tau : volt
dI/dt=(x-I)/tau1 : volt
dx/dt=-x/tau2 : volt
'''

T=-tau*log((I0-theta)/(I0-Vl))

P=NeuronGroup(N,model=eqs,threshold=theta,reset=Vl)
C=Connection(P,P,'x',weight=w)
c=1. # in 0..1
P.V=I0+(Vl-I0)*exp(-c*arange(N)/N*T/tau)

run(duration1)

M=PopulationRateMonitor(P) # average spiking activity
S=StateMonitor(P,'V') # to measure the individual variances of V

mv=[]
@network_operation
def monitor_averageV():
    mv.append(mean(P.V))

run(duration2)
deltaN=var(mv)
delta=mean(S.var)
sigma=deltaN/delta # Measure of synchrony
print sigma

# Visual check
plot(M.times/ms,M.smooth_rate(1*ms))
show()
