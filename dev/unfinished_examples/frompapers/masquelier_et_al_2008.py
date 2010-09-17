'''
This program duplicates the results of Masquelier et al. 2008 - STDP finds the
start of repeating patterns in continuous spike trains.

1000 neurons are connected to a single neuron by STDP synapses. Every 200ms,
the pattern RANDOM 100ms, PATTERN 50ms RANDOM 50ms is repeated, where RANDOM
is Poisson noise, and PATTERN is the same pattern repeated. More exactly,
half the neurons are patterned, and half are random all the time.

The STDP model is standard exponential STDP, not the one from that paper, and so
the results are slightly different here.
'''
from brian import *
from brian.utils.progressreporting import ProgressReporter
import time
import pylab

class TimedPoissonGroup(PoissonGroup):
    '''
    Fires Poisson spikes during a certain interval
    
    See documentation for :class:`PoissonGroup`, with the additional feature that
    spikes will only be fired between ``start`` and ``start+duration``. If ``period``
    is specified, it will repeat with that period.
    '''
    def __init__(self, N=1, rates=0*Hz, start=0*second, duration=1*second,
                 clock=None, period=None):
        end = float(start+duration)
        start = float(start)
        if period is not None:
            period = float(period)
        def f_rates(t):
            t = float(t)
            if period is not None:
                t = t%period
            if start<t<end:
                return rates
            else:
                return 0*Hz
        PoissonGroup.__init__(self, N, rates=f_rates, clock=clock)
        
def make_poisson_spiketrain(N, rates, start, duration, **kwds):
    '''
    Returns a spike train for N Poisson neurons firing at rates between start and duration
    '''
    c = Clock(**kwds)
    G = TimedPoissonGroup(N, rates, start, duration, c)
    M = SpikeMonitor(G)
    net = Network(G, M)
    net.run(start+duration)
    return M.spikes

# appears to be just about good enough, it's more stable if you make it smaller
defaultclock.dt = 0.25*ms

initial_segment_duration = 100*ms
pattern_segment_duration = 50*ms
end_segment_duration = 50*ms
N_unpatterned = 500 # Number of neurons that are occasionally patterned
N_patterned = 500   # Noise neurons
poprate = 50*Hz
repeats = 500

# The sequence repeats many times, we plot plotseq_n plots starting from
# the repeats in plotseq_startset, to show that the repeated part of the
# pattern looks different
plotseq_n = 5
plotseq_startset = [0, int(repeats/3), int((2*repeats)/3), repeats-plotseq_n]

# Parameters for STDP model
taum = 20*ms
tau_post = 20*ms
tau_pre = 20*ms
Ee = 0*mV
Vt = -54*mV
Vr = -60*mV
El = -70*mV
taue = 5*ms
gmax = 0.004           # Maximum synaptic strength
dA_pre = .005          # Potentiation rate
dA_post = -dA_pre*1.1  # Depression rate

total_duration = initial_segment_duration + pattern_segment_duration + end_segment_duration
N = N_unpatterned + N_patterned

# This is the spike train that will be repeated
pat_spikes = make_poisson_spiketrain(
                N_patterned, poprate, initial_segment_duration,
                pattern_segment_duration, dt=defaultclock.dt)

eqs_post = '''
dV/dt = (ge*(Ee-V)+El-V)/taum  : volt
dge/dt = -ge/taue              : 1
excitatory = ge
'''

# Poisson spikes for the patterned neurons for the initial segment (PG_start),
# for the end segment (PG_end), and for the unpatterned neurons throughout
# (PG_all), and the pattern for the patterned neurons in the middle segment
# (patgroup)
PG_start = TimedPoissonGroup(
            N_patterned, poprate, 0*ms,
            initial_segment_duration)
PG_end = TimedPoissonGroup(
            N_patterned, poprate,
            initial_segment_duration+pattern_segment_duration,
            end_segment_duration)
PG_all = TimedPoissonGroup(
            N_unpatterned, poprate, 0*ms,
            initial_segment_duration+pattern_segment_duration+end_segment_duration)
patgroup = SpikeGeneratorGroup(N_patterned, pat_spikes)

G_pre = NeuronGroup(N, 'V:1', threshold=0.5, reset=0.0)
G_unpatterned = G_pre.subgroup(N_unpatterned)
G_patterned = G_pre.subgroup(N_patterned)

C_start = IdentityConnection(PG_start, G_patterned)
C_end = IdentityConnection(PG_end, G_patterned)
C_all = IdentityConnection(PG_all, G_unpatterned)
C_pat = IdentityConnection(patgroup, G_patterned)

G_post = NeuronGroup(1, eqs_post, threshold=Vt, reset=Vr)

synapses = Connection(G_pre, G_post, 'excitatory', structure='dense')
synapses.connect(G_pre, G_post, rand(len(G_pre), len(G_post))*gmax)

stdp = ExponentialSTDP(synapses, tau_pre, tau_post, dA_pre, dA_post, wmax=gmax)

M_pre = SpikeMonitor(G_pre)
M_post = SpikeMonitor(G_post)
MV_post = StateMonitor(G_post, 'V', record=True)

G_post.V = Vr

net = MagicNetwork()

weights_before = array(synapses.W).copy()
t_start = time.time()

reporter = ProgressReporter('stderr')
for i in range(repeats):
    reporter.equal_subtask(i, repeats)
    reinit_default_clock()
    # Reinitialise these by hand because we don't want to reinitialise the
    # spike and state monitors
    PG_start.reinit()
    PG_end.reinit()
    PG_all.reinit()
    G_pre.reinit()
    G_post.reinit()
    patgroup.reinit()
    net.run(total_duration, report=reporter)

t_end = time.time()
weights_after = array(synapses.W).copy()

print 'Time taken:', t_end-t_start
print 'Spikes pre:', M_pre.nspikes
print 'Spikes post:', M_post.nspikes

figure()
subplot(221)
imshow(weights_before, interpolation='nearest', origin='lower', aspect='auto')
pylab.gray()
subplot(222)
imshow(weights_after, interpolation='nearest', origin='lower', aspect='auto')
pylab.gray()
subplot(223)
plot(weights_before.squeeze(),'.')
subplot(224)
plot(weights_after.squeeze(),'.')

figure()
V = MV_post[0]
n = int(len(V)/repeats)
for j, i in enumerate(plotseq_startset):
    subplot(2,2,j+1)
    title(str(i))
    for k in range(plotseq_n):
        l = i+k
        plot(V[l*n:l*n+n])

figure()
subplot(121)
i, t = zip(*M_post.spikes)
t = array(t)-initial_segment_duration
t[t<0*ms]=0*ms
t[t>pattern_segment_duration]=0*ms
latency_reduction_t = t.copy()
plot(latency_reduction_t,'.')
title('Latency reduction (fig 5 in paper)')
subplot(122)
i, t = zip(*M_post.spikes)
m = int(len(t)/20)
if m==0: m=1
t = array(t)-initial_segment_duration
s = [t[n:n+m] for n in range(0,len(t),m)]
ps = [float(sum((si>0*ms) & (si<pattern_segment_duration)))/len(si) for si in s]
plot(ps)
title('Fraction of spikes within pattern')

show()
