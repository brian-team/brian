"""
Van Rossum metric.
We express the metric as an STDP rule + counter.
"""
from brian import *

tau = 20 * ms
N = 100
b = 1.2 # constant current mean, the modulation varies
f = 10 * Hz
tau_VR=2*ms

eqs = '''
dv/dt=(-v+a*sin(2*pi*f*t)+b)/tau : 1
a : 1
u : 1 # fake variable
'''

neurons = NeuronGroup(N, model=eqs, threshold=1, reset=0)
neurons.v = rand(N)
neurons.a = linspace(.05, 0.75, N)
S = SpikeMonitor(neurons)
trace = StateMonitor(neurons, 'v', record=50)
VR=VanRossumMetric(neurons,tau=tau_VR)

# Alternative implementation
counter=SpikeCounter(neurons)
synapses=Connection(neurons,neurons,'u',weight=0,structure='dense')
m=1e6
stdp=ExponentialSTDP(synapses, taup=tau_VR, taum=tau_VR, Ap=1./m, Am=1./m, wmax=m)

run(1000 * ms)
subplot(221)
raster_plot(S)
subplot(222)
plot(trace.times / ms, trace[50])
subplot(223)
imshow(VR.get_distance())
subplot(224)
c=counter.count*ones((N,1))
M=c+c.T-2*synapses.W.todense()
imshow(M)

show()
