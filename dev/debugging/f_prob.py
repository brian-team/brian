from brian import *

# Works if you put this in:
#del f

eqs='''
dv/dt = I : 1
I = f : Hz
f : Hz
'''

neurons = NeuronGroup(1, eqs)
