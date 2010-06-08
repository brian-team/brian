from brian import *
eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''
P = NeuronGroup(40, model=eqs, threshold= -50 * mV, reset= -60 * mV)
P.v = -60 * mV + 10 * mV * rand(len(P))
Pe = P.subgroup(32)
Pi = P.subgroup(8)
Ce = Connection(Pe, P, 'ge')
Ci = Connection(Pi, P, 'gi')
Ce.connect_random(Pe, P, 1.0, weight=1.62 * mV)
Ci.connect_random(Pi, P, 0.02, weight= -9 * mV)
M = SpikeMonitor(P)
run(1 * ms)

# Here do the compress and count then number of connections
Ce.compress()
countCe = 0
for i in range(0, 32):
    for j in range(0, 41):
        if abs(Ce[i, j]) > 1e-10:
            countCe += 1

print Ce.W
print countCe
