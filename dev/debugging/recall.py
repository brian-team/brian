from brian import *

G = NeuronGroup(1, 'V:1')

@network_operation
def f():
    print '.'

run(.1*ms)

print 'first run over'

forget(f)

run(.1*ms)

print 'second run over'

recall(f)

run(.1*ms)

print 'third run over'