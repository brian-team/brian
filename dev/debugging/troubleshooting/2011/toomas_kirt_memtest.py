from brian import *
import gc
piecesize = 200
def break_up_connection(source, target, var, weight, sparseness, piecesize=piecesize):
    N = len(source)
    pieces = range(0, N, piecesize)+[N]
    conns = []
    for start, end in zip(pieces[:-1], pieces[1:]):
        C = Connection(source[start:end], target, var, weight=weight,
                       sparseness=sparseness,
                       column_access=True)
#        C.compress()
        gc.collect()
        conns.append(C)
#    magic_register(*conns)
    return conns


eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''
P = NeuronGroup(1000, eqs, threshold=-50*mV, reset=-60*mV)
P.v = -60*mV
Pe = P.subgroup(800)
Pi = P.subgroup(200)
Ce = break_up_connection(Pe, P, 'ge', weight=1.62*mV, sparseness=0.2)
Ci = break_up_connection(Pi, P, 'gi', weight=-9*mV, sparseness=0.2)

# Plasticity
gmax = 0.01
tau_pre = 20 * ms
tau_post = tau_pre
dA_pre = .01
dA_post = -dA_pre * tau_pre / tau_post * 2.5
stdp = ExponentialSTDP(Ce[0], tau_pre, tau_post, dA_pre, dA_post, wmax=gmax, update='mixed')

M = SpikeMonitor(P)

run(1*second,report='text')
raster_plot(M)
show()
