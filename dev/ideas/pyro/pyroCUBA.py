'''
Very short example program.
'''
from brian import *
import Pyro.core
from Pyro.errors import PyroError


class BrianServer(Pyro.core.ObjBase):
    def __init__(self):
        Pyro.core.ObjBase.__init__(self)

Pyro.core.initServer()
daemon = Pyro.core.Daemon()
print
print 'The Pyro Daemon is running on ', daemon.hostname + ':' + str(daemon.port)
print '(you may need this info for the client to connect to)'
print

eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

P = NeuronGroup(4000, model=eqs,
              threshold= -50 * mV, reset= -60 * mV)
P.v = -60 * mV + 10 * mV * rand(len(P))
Pe = P.subgroup(3200)
Pi = P.subgroup(800)

Ce = Connection(Pe, P, 'ge', weight=1.62 * mV, sparseness=0.02)
Ci = Connection(Pi, P, 'gi', weight= -9 * mV, sparseness=0.02)

M = SpikeMonitor(P)

# Pyro loop
@network_operation
def server():
    pass

run(1 * second)
raster_plot(M)
show()
