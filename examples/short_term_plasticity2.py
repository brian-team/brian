'''
Network (CUBA) with depressing synapses
Excitatory synapses are depressing, inhibitory ones are not
'''
from brian import *

U_SE=.2
tau_rec=500*ms

eqs='''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
dR/dt=(1-R)/tau_rec : 1 # synaptic resource (in [0,1])
'''

def reset_STP(P,spikes):
    P.R_[spikes]-=U_SE*P.R_[spikes]
    P.v_[spikes]=-60*mV

P=NeuronGroup(4000,model=eqs,threshold=-50*mV,reset="R-=U_SE*R;v=-60*mV")
P.v=-60*mV+rand(4000)*10*mV
P.R=1
Pe=P.subgroup(3200)
Pi=P.subgroup(800)
Ce=Connection(Pe,P,'ge',modulation='R')
Ci=Connection(Pi,P,'gi')
Ce.connect_random(Pe, P, 0.02,weight=1.62*mV)
Ci.connect_random(Pi, P, 0.02,weight=-9*mV)
M=SpikeMonitor(P)
trace=StateMonitor(P,'R',record=0)
rate=PopulationRateMonitor(P)
run(1*second)
print M.nspikes,"spikes"
subplot(311)
raster_plot(M)
subplot(312)
plot(trace.times/ms,trace[0])
subplot(313)
plot(rate.times/ms,rate.smooth_rate(5*ms))
show()
