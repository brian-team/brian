from brian import *
import gc
piecesize = 1000
def break_up_connection(source, target, var, weight, sparseness, piecesize=piecesize):
    N = len(source)
    pieces = range(0, N, piecesize)+[N]
    conns = []
    for start, end in zip(pieces[:-1], pieces[1:]):
        C = Connection(source[start:end], target, var, weight=weight,
                       sparseness=sparseness,
                       column_access=False)
        C.compress()
        gc.collect()
        conns.append(C)
    magic_register(*conns)
    return conns
eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''
P = NeuronGroup(11000, eqs, threshold=-50*mV, reset=-60*mV)
P.v = -60*mV
Pe = P.subgroup(8500)
Pi = P.subgroup(2500)
Ce = break_up_connection(Pe, P, 'ge', weight=1.62*mV, sparseness=0.2)
Ci = break_up_connection(Pi, P, 'gi', weight=-9*mV, sparseness=0.2)
M = SpikeMonitor(P)

run(1*second)
raster_plot(M)
show()
