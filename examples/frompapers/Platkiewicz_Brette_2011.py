#!/usr/bin/env python
'''
Slope-threshold relationship with noisy inputs, in the adaptive threshold model
-------------------------------------------------------------------------------
Fig. 5E,F from:
Platkiewicz J and R Brette (2011). Impact of Fast Sodium Channel Inactivation on Spike
Threshold Dynamics and Synaptic Integration. PLoS Comp Biol 7(5):
e1001129. doi:10.1371/journal.pcbi.1001129
'''
from brian import *
from scipy import stats,optimize
from scipy.stats import linregress

rectify=lambda x:clip(x/volt,0,Inf)*volt

N=200 # 200 neurons to get more statistics, only one is shown
duration=1*second
#--Biophysical parameters
ENa=60*mV
EL=-70*mV
vT=-55*mV
Vi=-63*mV
tauh=5*ms
tau=5*ms
ka=5*mV
ki=6*mV
a=ka/ki
tauI=5*ms
mu=15*mV
sigma=6*mV/sqrt(tauI/(tauI+tau))

#--Theoretical prediction for the slope-threshold relationship (approximation: a=1+epsilon)
thresh=lambda slope,a: Vi-slope*tauh*log(1+(Vi-vT/a)/(slope*tauh))
#-----Exact calculation of the slope-threshold relationship
thresh_ex=lambda s: optimize.fsolve(lambda th: a*s*tauh*exp((Vi-th)/(s*tauh))-th*(1-a)-a*(s*tauh+Vi)+vT,thresh(s,a))*volt

eqs="""
dv/dt=(EL-v+mu+sigma*I)/tau : volt
dtheta/dt=(vT+a*rectify(v-Vi)-theta)/tauh : volt
dI/dt=-I/tauI+(2/tauI)**.5*xi : 1 # Ornstein-Uhlenbeck
"""
neurons=NeuronGroup(N,eqs,threshold="v>theta",reset='v=EL',refractory=5*ms)
neurons.v=EL
neurons.theta=vT
neurons.I=0
S=SpikeMonitor(neurons)
M=StateMonitor(neurons,'v',record=True)
Mt=StateMonitor(neurons,'theta',record=0)

run(duration,report='text')

# Linear regression gives depolarization slope before spikes
tx=M.times[(M.times>0) & (M.times<1.5*tauh)]
slope,threshold=[],[]
v=array(M._values)
for (i,t) in S.spikes:
    ind=(M.times<t) & (M.times>t-tauh)
    mx=v[:,i][ind]
    s,_,_,_,_=linregress(tx[:len(mx)],mx)
    slope.append(s)
    threshold.append(mx[-1])

# Figure
M.insert_spikes(S) # displays spikes on the trace
subplot(221)
ind=M.times<500*ms
plot(M.times[ind]/ms,M[0][ind]/mV,'k')
plot(Mt.times[ind]/ms,Mt[0][ind]/mV,'r')
xlabel('Time (ms)')
ylabel('Voltage (mV)')

subplot(222)
plot(slope,array(threshold)/mV,'r.')
sx=linspace(0.5*volt/second,4*volt/second,100)
t=array([thresh_ex(s*volt/second) for s in sx])
plot(sx,t/mV,'k')
xlim(0.5,4)
xlabel('Depolarization slope (ms)')
ylabel('Threshold (mV/ms)')

show()
