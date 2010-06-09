from brian import *

rem = []

def f():
    G = NeuronGroup(1, 'V:1')
    rem.append(G)

f()

clear(True, all=True)
#clear(True)

print rem[0]._S
