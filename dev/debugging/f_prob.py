from brian import *

# Works if you put this in:
#del f

eqs='''
dv/dt = I : 1
f : Hz
I = f : Hz
'''

neurons = NeuronGroup(1, eqs)
