from brian import *

G = NeuronGroup(1, 'dV/dt = xi/(10*ms)**0.5 : 1')
MR = RecentStateMonitor(G, 'V')
M = StateMonitor(G, 'V', record=True)
run(7*ms)

print MR.get_past_values(array([0*ms]))
print MR.get_past_values(array([1*ms]))
print MR.get_past_values(array([2*ms]))
print MR.get_past_values_sequence([array([0*ms]), array([1*ms]), array([2*ms])])

M.plot()
plot(MR.times, MR[0]+0.1)
MR.plot()
show()