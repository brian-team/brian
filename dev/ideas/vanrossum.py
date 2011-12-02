"""
Van Rossum metric.
We express the metric as an STDP rule + counter.

I think for Poisson inputs we get (Fi+Fj)*tau/2 (after normalization by the
duration).
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
'''

neurons = NeuronGroup(N, model=eqs, threshold=1, reset=0)
neurons.v = rand(N)
neurons.a = linspace(.05, 0.75, N)
S = SpikeMonitor(neurons)
trace = StateMonitor(neurons, 'v', record=50)
VR=VanRossumMetric(neurons,tau=tau_VR)

# Alternative implementation
class NewVanRossumMetric(STDP):
    def __init__(self,source,tau=2*ms):    
        self.counter=SpikeCounter(neurons)
        self.N=len(neurons)
        class Fake(object):
            def __init__(self,source):
                self.source=source
                self.target=source
                self.delay=0
                self.W=zeros((len(source),len(source)))
        self.synapses=Fake(neurons)
        
        eqs_stdp='''
        dApost/dt=-Apost/tau_VR : 1
        dApre/dt=-Apre/tau_VR : 1
        '''
        STDP.__init__(self,self.synapses,eqs_stdp,pre='Apre+=1;w+=Apost',post='Apost+=1;w+=Apre')
        self.contained_objects.append(self.counter)
        
    def get_distance(self):
        c=self.counter.count*ones((self.N,1))
        return c+c.T-2*self.synapses.W # divide by duration?

VR2=NewVanRossumMetric(neurons,tau=tau_VR)

run(1000 * ms)
subplot(221)
raster_plot(S)
subplot(222)
plot(trace.times / ms, trace[50])
subplot(223)
imshow(VR.get_distance())
subplot(224)
imshow(VR2.get_distance())

show()
