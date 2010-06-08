'''
Polychronization: Computation with Spikes
Eugene M. Izhikevich
Neural Computation, 2006

See Izhikevich_2006_Polychronization.py for an explanation of this code.

In this version, we use various tricks to make the simulation much faster,
including inline C++ code. We also save the progress as the simulation runs,
and analyse it at the end for polychronous groups.

Note that you will need to create two folders data and imgs to run this
script.
'''
import brian_no_units_no_warnings
from brian import *
import random
from scipy import weave
import cPickle as pickle

try:
    imported_data = pickle.load(open('data/izhikevich.pickle', 'rb'))
    defaultclock.t = imported_data['t']
    print 'Starting from saved progress at time', defaultclock.t
except IOError:
    imported_data = {}

dt = defaultclock.dt = 1 * ms

M = 100                # number of synapses per neuron
D = 20 * ms              # maximal conduction delay
D_min = 1 * ms           # minimal conduction delay
D_i = 1 * ms             # inhibitory neuron conduction delay

# excitatory neurons
Ne = 800
a_e = 0.02 / ms
d_e = 8 * mV / ms
s_e = 6 * mV / ms
# inhibitory neurons
Ni = 200
a_i = 0.1 / ms
d_i = 2 * mV / ms
s_i = -5 * mV / ms
# all neurons
b = 0.2 / ms
c = -65 * mV
sm = 10 * mV / ms             # maximal synaptic strength
N = Ne + Ni

thalamic_input_weight = 20 * mV / ms

eqs = Equations('''
dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u   : volt
du/dt = a*(b*v-u)                                : volt/second
a                                                : 1/second
d                                                : volt/second
I                                                : volt/second
''')
reset = '''
v = c
u += d
'''
threshold = 30 * mV

G = NeuronGroup(N, eqs, threshold=threshold, reset=reset)

def make_Izhikevich_scheme():
    c1 = 0.04 / ms / mV
    c2 = 5 / ms
    c3 = 140 * mV / ms
    # Izhikevich's numerical integration scheme
    def Izhikevich_scheme(G):
        G.v += (0.5 * dt) * (c1 * G.v ** 2 + c2 * G.v + c3 - G.u + G.I)
        G.v += (0.5 * dt) * (c1 * G.v ** 2 + c2 * G.v + c3 - G.u + G.I)
        G.u += dt * G.a * (b * G.v - G.u)
    weave_code = '''
    for(int i=0; i<N; i++){
        v[i] += (0.5*dt)*(c1*v[i]*v[i]+c2*v[i]+c3-u[i]+I[i]);
        v[i] += (0.5*dt)*(c1*v[i]*v[i]+c2*v[i]+c3-u[i]+I[i]);
        u[i] += dt*a[i]*(b*v[i]-u[i]);
    }
    '''
    weave_vars = (G.a, G.v, G.u, G.I, c1, c2, c3)
    weave_compiler = get_global_preference('weavecompiler')
    extra_compile_args = ['-O3']
    if weave_compiler == 'gcc':
        extra_compile_args += get_global_preference('gcc_options') # ['-march=native', '-ffast-math']
    def weave_Izhikevich_scheme(G):
        a, v, u, I, c1, c2, c3 = weave_vars
        weave.inline(weave_code, ['a', 'v', 'u', 'I', 'c1', 'c2', 'c3', 'b', 'dt', 'N'],
                     compiler=weave_compiler, extra_compile_args=extra_compile_args)
    if get_global_preference('useweave'):
        return weave_Izhikevich_scheme
    return Izhikevich_scheme
G._state_updater = make_Izhikevich_scheme()

Ge = G.subgroup(Ne)
Gi = G.subgroup(Ni)

G.v = c
G.u = b * c
Ge.a = a_e
Ge.d = d_e
Gi.a = a_i
Gi.d = d_i

zeromVms = 0 * mV / ms
@network_operation(when='before_connections')
def thalamic_input():
    G.I = zeromVms
    G.I[randint(N)] = float(thalamic_input_weight)

if imported_data:
    G._S[:] = imported_data['G._S']

Ce = Connection(Ge, G, 'I', delay=True, max_delay=D)
Ci = Connection(Gi, Ge, 'I', delay=D_i)
Ce_deriv = Connection(Ge, G, 'I', delay=True, max_delay=D)
forget(Ce_deriv)

if imported_data:
    exc_inds = imported_data['exc_inds']
    exc_weights = imported_data['exc_weights']
    exc_delays = imported_data['exc_delays']
    inh_inds = imported_data['inh_inds']
    inh_weights = imported_data['inh_weights']
    for i in xrange(Ne):
        Ce[i, exc_inds[i]] = exc_weights[i]
        Ce.delay[i, exc_inds[i]] = exc_delays[i]
        Ce_deriv[i, exc_inds[i]] = exc_weights[i]
        Ce_deriv.delay[i, exc_inds[i]] = exc_delays[i]
    for i in xrange(Ni):
        Ci[i, inh_inds[i]] = inh_weights[i]
else:
    for i in xrange(Ne):
        inds = random.sample(xrange(N), M)
        Ce[i, inds] = s_e * ones(len(inds))
        Ce.delay[i, inds] = rand(len(inds)) * (D - D_min) + D_min
    for i in xrange(Ni):
        inds = random.sample(xrange(Ne), M)
        Ci[i, inds] = s_i * ones(len(inds))
    for i in xrange(Ne):
        Ce_deriv[i, :] = Ce[i, :]
        Ce_deriv.delay[i, :] = Ce.delay[i, :]

artificial_wmax = 1e10 * mV / ms
stdp = ExponentialSTDP(Ce_deriv,
                       taup=20 * ms, taum=20 * ms,
                       Ap=(0.1 * mV / ms / artificial_wmax),
                       Am=(-0.12 * mV / ms / artificial_wmax),
                       wmin= -artificial_wmax,
                       wmax=artificial_wmax,
                       interactions='nearest'
                       )

Ce_deriv.compress()
Ce_deriv.W.alldata[:] = 0
weight_derivative_const = 0.01 * mV / ms
@network_operation(clock=EventClock(dt=1 * second))
def update_weights_from_derivative():
    Ce.W.alldata[:] = clip(Ce.W.alldata + weight_derivative_const + Ce_deriv.W.alldata, 0, sm)
    Ce_deriv.W.alldata[:] *= 0.9

M = SpikeMonitor(G)

def autocorrelogram(T, width=40 * ms, bin=1 * ms):
    b = bincount(array(array(T) / bin, dtype=int))
    ac = correlate(b, b, mode='full')
    ac = ac[len(ac) / 2 - int(width / bin):len(ac) / 2 + int(width / bin) + 1]
    edges = arange(-width, width + bin, bin)
    return ac, edges

@network_operation(clock=EventClock(dt=10 * second, t=defaultclock.t + 10 * second))
def plot_recent_output():
    if not M.spikes:
        return
    clf()
    subplot(211)
    raster_plot(M, ms=1)
    s = str(int((defaultclock.t + .5 * ms) / second))
    title('t = ' + s + ' s')
    subplot(212)
    ac, edges = autocorrelogram([t for i, t in M.spikes], width=.3 * second, bin=5 * ms)
    emin = min((edges > 1. / 100).nonzero()[0])
    plot(1 / edges[emin:], ac[emin:])
    gcf().savefig('imgs/iz' + s + '.png')
    pickle.dump(M.spikes, open('data/iz_spikes' + s + '.pickle', 'wb'), -1)
    M.reinit()
    print 'Plotted up to time', s

@network_operation(clock=EventClock(dt=100 * second, t=defaultclock.t + 100 * second))
def save_progress():
    s = str(int((defaultclock.t + .5 * ms) / second))
    imported_data['G._S'] = G._S
    imported_data['t'] = defaultclock.t
    exc_inds = [Ce.W[i, :].ind for i in range(Ne)]
    exc_weights = [asarray(Ce.W[i, :]) for i in range(Ne)]
    exc_delays = [asarray(Ce.delay[i, :]) for i in range(Ne)]
    inh_inds = [Ci.W[i, :].ind for i in range(Ni)]
    inh_weights = [asarray(Ci.W[i, :]) for i in range(Ni)]
    imported_data['exc_inds'] = exc_inds
    imported_data['exc_weights'] = exc_weights
    imported_data['exc_delays'] = exc_delays
    imported_data['inh_inds'] = inh_inds
    imported_data['inh_weights'] = inh_weights
    pickle.dump(imported_data, open('data/izhikevich.pickle', 'wb'), -1)
    print 'Saved progress up to time', s

run(3600 * second - defaultclock.t, report='stderr')

