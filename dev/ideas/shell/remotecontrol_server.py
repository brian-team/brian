from brian import *
from brian.experimental.remotecontrol import *

eqs='''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''
P=NeuronGroup(4000,model=eqs,
              threshold=-50*mV,reset=-60*mV)
P.v=-60*mV+10*mV*rand(len(P))
Pe=P.subgroup(3200)
Pi=P.subgroup(800)
Ce=Connection(Pe,P,'ge',weight=1.62*mV,sparseness=0.02)
Ci=Connection(Pi,P,'gi',weight=-9*mV,sparseness=0.02)
M=SpikeMonitor(P)
@network_operation(clock=EventClock(dt=1*second))
def clearspikes():
    M.reinit()
    print 'Simulated', defaultclock.t

srv = remote_control_server()

run(1e10*second)

print 'Finished!'
