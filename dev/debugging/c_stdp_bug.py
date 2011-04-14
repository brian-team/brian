'''
There's a bug in CSTDP so that if you have more target neurons than sources,
it will segfault, suggesting that one of the values passed to the C code is
wrong.
'''
from brian import *

set_global_preferences(useweave=True, usecstdp=True)
log_level_debug()

Nin = 100
Nout = 200
duration = 40 * second

Fmax = 50 * Hz

tau = 10 * ms
taue = 2 * ms
taui = 5 * ms
sigma = 0.#0.4
# Note that factor (tau/taue) makes integral(v(t)) the same when the connection
# acts on ge as if it acted directly on v.
eqs = Equations('''
#dv/dt = -v/tau + sigma*xi/(2*tau)**.5 : 1
dv/dt = (-v+(tau/taue)*ge-(tau/taui)*gi)/tau + sigma*xi/(2*tau)**.5 : 1
dge/dt = -ge/taue : 1
dgi/dt = -gi/taui : 1
excitatory = ge
inhibitory = gi
''')
reset = 0
threshold = 1
refractory = 0 * ms
taup = 5 * ms
taud = 5 * ms
Ap = .1
Ad = -Ap * taup / taud * 1.2
wmax_ff = 0.1
wmax_rec = wmax_ff
wmax_inh = wmax_rec

width = 0.2

recwidth = 0.2

Gin = PoissonGroup(Nin)
Gout = NeuronGroup(Nout, eqs, reset=reset, threshold=threshold, refractory=refractory)
ff = Connection(Gin, Gout, 'excitatory', structure='dense')
for i in xrange(Nin):
    ff[i, :] = (rand(Nout) > .5) * wmax_ff
rec = Connection(Gout, Gout, 'excitatory')
for i in xrange(Nout):
    d = abs(float(i) / Nout - linspace(0, 1, Nout))
    d[d > .5] = 1. - d[d > .5]
    dsquared = d ** 2
    prob = exp(-dsquared / (2 * recwidth ** 2))
    prob[i] = -1
    inds = (rand(Nout) < prob).nonzero()[0]
    w = rand(len(inds)) * wmax_rec
    rec[i, inds] = w

inh = Connection(Gout, Gout, 'inhibitory', sparseness=1, weight=wmax_inh)

stdp_ff = ExponentialSTDP(ff, taup, taud, Ap, Ad, wmax=wmax_ff)
#stdp_rec = ExponentialSTDP(rec, taup, taud, Ap, Ad, wmax=wmax_rec)

run(0 * ms)
@network_operation(clock=EventClock(dt=20 * ms))
def stimulation():
    Gin.rate = Fmax * exp(-(linspace(0, 1, Nin) - rand())**2 / (2 * width ** 2))

run(1*second, report='stderr')
