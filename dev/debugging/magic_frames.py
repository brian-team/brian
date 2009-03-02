from brian import *
import gc

do_collect = False
do_permanent_store = False

ps = []

def f():
    G = NeuronGroup(1, 'V:1')
    if do_permanent_store: ps.append(G)
    print 'id(G) =', id(G)
    print map(id, MagicNetwork().groups)
    
f()
if do_collect: gc.collect()
f()
if do_collect: gc.collect()
f()
if do_collect: gc.collect()
f()