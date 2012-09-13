#!/usr/bin/env python
'''
Peter Diehl's entry for the 2012 Brian twister.
'''
from brian import *
eqs = '''
dv/dt = ((-60.*mV-v)+(I_synE+I_synI+I_b)/(10.*nS))/(20*ms)  : volt
I_synE =  3.*nS*ge*(  0.*mV-v)                              : amp
I_synI = 30.*nS*gi*(-80.*mV-v)                              : amp
I_b                                                         : amp
dge/dt = -ge/( 5.*ms)                                       : 1
dgi/dt = -gi/(10.*ms)                                       : 1
'''
P = NeuronGroup(10000, eqs, threshold=-50.*mV, refractory=5.*ms, reset=-60.*mV)
Pe = P.subgroup(8000)
Pi = P.subgroup(2000)
Ce  = Connection(Pe, P,  'ge', weight=1., sparseness=0.02)
Cie = Connection(Pi, Pe, 'gi', weight=1., sparseness=0.02)
Cii = Connection(Pi, Pi, 'gi', weight=1., sparseness=0.02)
eqs_stdp = '''
dpre/dt  =  -pre/(20.*ms)         : 1.0
dpost/dt = -post/(20.*ms)         : 1.0
'''
nu = 0.1              # learning rate
alpha = 0.12          # controls the firing rate
stdp = STDP(Cie, eqs=eqs_stdp, pre='pre+= 1.; w+= nu*(post-alpha)', 
            post='post+= 1.; w+= nu*pre', wmin=0., wmax= 10.)
M = PopulationRateMonitor(Pe, bin = 1.)
P.I_b = 200.*pA       #set the input current
run(10*second)
P.I_b = 600.*pA       #increase the input and see how the rate adapts
run(10*second) 
plot(M.times[0:-1]/second, M.rate[0:-1])
show()