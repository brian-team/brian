'''
Spike-timing dependent plasticity
Adapted from Fig. 3 in Song and Abbott (2001)

About 10 times slower than real time.

'''
from brian import *
from time import time

Nin = 1000
Nout = 200
taum = 20 * ms
tau_pre = 20 * ms
tau_post = tau_pre
Ee = 0 * mV
vt = -54 * mV
vr = -60 * mV
vrest = -74 * mV
taue = 5 * ms
gmax = .02
dA_pre = .001*5
dA_post_ff = -dA_pre * tau_pre / tau_post * 1.06
dA_post_rec = -dA_pre * tau_pre / tau_post * 1.04
R0=10*Hz
R1=80*Hz
sigma=100.
stimulus_period=20*ms
Fext=150*Hz
gext=0.096

input_rate=lambda s,a:R0+R1*(exp(-(s-a)**2/(2.*sigma**2))+exp(-(s+1000.-a)**2/(2.*sigma**2))+exp(-(s-1000.-a)**2/(2.*sigma**2)))
all_inputs=arange(Nin)

eqs_neurons = '''
dv/dt=(ge*(Ee-v)+vrest-v)/taum : volt   # the synaptic current is linearized
dge/dt=-ge/taue : 1
'''

input = PoissonGroup(Nin, rates=input_rate(all_inputs,500))
input_ext=PoissonGroup(Nout,rates=Fext)
neurons = NeuronGroup(Nout, model=eqs_neurons, threshold=vt, reset=vr)
synapses_ff = Connection(input, neurons, 'ge', weight=lambda: rand()*gmax, sparseness=0.2)
#synapses_ff.W.alldata=rand(synapses_ff.W.nnz)*gmax
#synapses_rec = Connection(neurons, neurons, 'ge', weight=0, structure='dense')
synapses_ext = IdentityConnection(input_ext,neurons,'ge',weight=gext)
neurons.v = vrest

stdp_ff=ExponentialSTDP(synapses_ff,tau_pre,tau_post,dA_pre,dA_post_ff,wmax=gmax)
#stdp_rec=ExponentialSTDP(synapses_rec,tau_pre,tau_post,dA_pre,dA_post_rec,wmax=gmax)

next_change=exponential()*stimulus_period # maybe define a new clock?

@network_operation
def new_input(cl):
    global next_change
    if cl.t>=next_change: # new input
        input.rate=input_rate(all_inputs,randint(Nin))
        next_change=exponential()*stimulus_period

rate = PopulationRateMonitor(neurons)

start_time = time()
run(30 * second, report='text')
print "Simulation time:", time() - start_time

figure()
subplot(311)
plot(rate.times / second, rate.smooth_rate(100 * ms))
subplot(312)
hist(synapses_ff.W.alldata / gmax, 20)

figure()
imshow(synapses_ff.W.todense()/gmax)
colorbar()

#figure()
#imshow(synapses_rec.W.todense()/gmax)
#colorbar()

show()
