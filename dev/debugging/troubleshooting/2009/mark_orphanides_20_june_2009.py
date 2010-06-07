from brian import *
#G = NeuronGroup(1, 'v:1\nw:1', threshold='(v>1)&(w>1)')
#M = SpikeMonitor(G)
#G.v = 2
#run(1*ms)
#print M.spikes
#G.w = 2
#run(1*ms)
#print M.spikes

eqs='''
dV/dt = dV_dt: 1
dV_dt = V/(3*ms) : 1/second
thresh = (V>1)&(dV_dt>1/second) : 1
'''

G=NeuronGroup(1, eqs, threshold=EmpiricalThreshold(0, 1*ms, 'thresh'))
G.V=0.1
M=MultiStateMonitor(G, ('V', 'dV_dt', 'thresh'), record=True)

run(10*ms)

subplot(311)
M['V'].plot()
subplot(312)
M['dV_dt'].plot()
subplot(313)
M['thresh'].plot()
show()
