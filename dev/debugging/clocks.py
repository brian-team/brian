from brian import *

G = NeuronGroup(1, "V:1")

print id(defaultclock)

k = Clock(dt=2*ms)

print id(k)

#@network_operation(Clock(dt=2*ms))
@network_operation(k)
def netop():
    pass

print id(defaultclock)

# no problem if this monitor is defined BEFORE network operation
M = StateMonitor(G, 'V', record=True) 

run(10*ms)
print len(M.values[0]) # prints 5 instead of 101 
