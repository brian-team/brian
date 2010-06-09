'''
Still to do:

* Estimate with and without compiling turned on
* Investigate and estimate the cost of cache misses in
  the spiking part
'''
from brian import *
from run_opts import *
import time
import scipy
import numpy

def network_and_lazy_group(N=1):
    G = [NeuronGroup(1, model='dV/dt=-V/(10*ms):volt') for _ in range(N)]
    net = Network(G)
    net.run(0 * ms)
    #print len(net._update_schedule[id(defaultclock)])
    reinit_default_clock()
    t = time.time()
    net.run(duration)
    return time.time() - t

def network_lazy_group_and_zero_connections(N=1):
    G = NeuronGroup(1, model='dV/dt=-V/(10*ms):volt')
    C = [ Connection(G, G) for _ in range(N) ]
    net = Network(G, C)
    # problem with MultiConnection, this reproduces part of the activity of
    # network.prepare so that MultiConnection isn't used
    net.set_clock()
    net.connections = C
    for c in C:
        c.compress()
    net._build_update_schedule()
    net.prepared = True
    net.run(0 * ms)
    #print len(net._update_schedule[id(defaultclock)])
    reinit_default_clock()
    t = time.time()
    net.run(duration)
    return time.time() - t

def linear_network(N=100, vars=None):
    if vars is None:
        vars = 3
        eqs = '''
        dV/dt = (ge+gi-(V+49*mV))/(20*ms) : volt
        dge/dt = -ge/(5*ms) : volt
        dgi/dt = -gi/(10*ms) : volt
        '''
    else:
        eqs = ''
        for i in range(vars):
            # upper triangular differential equation
            eqs += 'dV' + str(i) + '/dt=-(' + '+'.join('V' + str(j) for j in range(i + 1)) + ')/(10*ms):volt\n'
    G = NeuronGroup(N, model=eqs)
    net = Network(G)
    net.run(0 * ms)
    reinit_default_clock()
    t = time.time()
    net.run(duration)
    return time.time() - t

def linear_network_cachemiss(N=100, vars=None):
    if vars is None:
        vars = 3
        eqs = '''
        dV/dt = (ge+gi-(V+49*mV))/(20*ms) : volt
        dge/dt = -ge/(5*ms) : volt
        dgi/dt = -gi/(10*ms) : volt
        '''
    else:
        eqs = ''
        for i in range(vars):
            # upper triangular differential equation
            eqs += 'dV' + str(i) + '/dt=-(' + '+'.join('V' + str(j) for j in range(i + 1)) + ')/(10*ms):volt\n'
    G = NeuronGroup(N, model=eqs)
    x = numpy.ones(cachemiss_array_size)
    @network_operation
    def hitcache():
        x[:] = 1.
    net = Network(G, hitcache)
    net.run(0 * ms)
    reinit_default_clock()
    t = time.time()
    net.run(duration)
    return time.time() - t

def spiking_network(rate, synapses_per_neuron=1):
    # rate is per neuron firing rate
    eqs = 'dV/dt=-V/(1*Msecond):volt'
    rate = int(N_spiking_network * rate * defaultclock.dt)
    G = NeuronGroup(N_spiking_network, model=eqs, threshold=10 * mV, reset=1 * volt)
    H = NeuronGroup(synapses_per_neuron, model='V:volt')
    G.V[:rate] = 1 * volt
    G.V[rate:] = 0 * volt
    C = Connection(G, H, 'V')
    C.connect_full(G, H, weight=1 * nvolt)
    net = Network(G, H, C)
    net.run(0 * ms)
    reinit_default_clock()
    t = time.time()
    net.run(duration)
    return time.time() - t

def threshold(N=100, abovethresh=50):
    G = NeuronGroup(N, model='V:volt', threshold=1 * mV)
    G.V[:abovethresh] = 2 * mV
    net = Network(G)
    net.run(0 * ms)
    reinit_default_clock()
    t = time.time()
    net.run(duration)
    return time.time() - t

def threshold_cachemiss(N=100, abovethresh=50):
    G = NeuronGroup(N, model='V:volt', threshold=1 * mV)
    G.V[:abovethresh] = 2 * mV
    x = numpy.ones(cachemiss_array_size)
    @network_operation
    def hitcache():
        x[:] = 1.
    net = Network(G, hitcache)
    net.run(0 * ms)
    reinit_default_clock()
    t = time.time()
    net.run(duration)
    return time.time() - t

def do_time(f, *params):
    return (f(*params) * second) / (duration / defaultclock.dt)

time_network_and_lazy_group = [ do_time(network_and_lazy_group, i) for i in range(1, max_groups + 1) ]
slope, intercept, r, tt, stderr = scipy.stats.linregress(range(1, max_groups + 1), time_network_and_lazy_group)
overhead_network = intercept * second
overhead_neurongroup = slope * second
print 'Network update overhead per dt =', overhead_network
print 'NeuronGroup overhead per dt (approx, with 1 linear neuron) =', overhead_neurongroup
subplot(331)
plot(range(1, max_groups + 1), time_network_and_lazy_group)
plot(range(0, max_groups + 1), [intercept + i * slope for i in range(0, max_groups + 1)])
title('Groups overhead')

time_network_lazy_group_and_zero_connections = [ do_time(network_lazy_group_and_zero_connections, i) for i in range(max_connections + 1) ]
slope, intercept, r, tt, stderr = scipy.stats.linregress(range(max_connections + 1), time_network_lazy_group_and_zero_connections)
overhead_connection = slope * second
print 'Connection overhead per dt =', overhead_connection
print 'Sanity check, these should be roughly equal:', intercept * second, ',', overhead_network + overhead_neurongroup
subplot(332)
plot(range(max_connections + 1), time_network_lazy_group_and_zero_connections)
plot(range(max_connections + 1), [intercept + i * slope for i in range(max_connections + 1)])
title('Connections overhead')

time_linear_network = [ do_time(linear_network, N) for N in N_linear_network ]
slope, intercept, r, tt, stderr = scipy.stats.linregress(N_linear_network, time_linear_network)
perneuron = slope * second
print 'Sanity check, these should be roughly equal:', intercept * second, ',', overhead_network + overhead_neurongroup
print 'Time per linear neuron with numstates=3 per dt =', perneuron
subplot(333)
plot(N_linear_network, time_linear_network)
plot([0] + N_linear_network, [intercept + i * slope for i in [0] + N_linear_network])
title('Linear 3 states')
xlabel('N')

time_linear_network_cachemiss = [ do_time(linear_network_cachemiss, N) for N in N_linear_network ]
slope, intercept, r, tt, stderr = scipy.stats.linregress(N_linear_network, time_linear_network_cachemiss)
perneuron_cachemiss = slope * second
print 'Time per linear neuron with numstates=3 per dt, cachemiss:', perneuron_cachemiss
subplot(334)
plot(N_linear_network, time_linear_network_cachemiss)
plot([0] + N_linear_network, [intercept + i * slope for i in [0] + N_linear_network])
title('Linear 3 states cachemiss')
xlabel('N')

#time_multivar_linear_network = [ do_time(linear_network, N_multivar_linear_network, M) for M in vars_linear_network ]
#fitcoeffs = polyfit(vars_linear_network, time_multivar_linear_network, 2)
#qfitcoeffs = array(fitcoeffs)*second
#perneuron_variableconst = slope*second
#print 'Time per linear neuron as function of numstates=M:', qfitcoeffs[1]/N_multivar_linear_network, 'M +', qfitcoeffs[0]/N_multivar_linear_network, 'M^2' 
#subplot(334)
#plot(vars_linear_network, time_multivar_linear_network)
#plot([0]+vars_linear_network, [fitcoeffs[2]+i*fitcoeffs[1]+i*i*fitcoeffs[0] for i in [0]+vars_linear_network])
#title('Linear in numstates, numneurons='+str(N_multivar_linear_network))

time_spiking_network = [ do_time(spiking_network, rate, fixed_syn_spiking_network) for rate in rates_spiking_network ]
slope, intercept, r, tt, stderr = scipy.stats.linregress(rates_spiking_network, time_spiking_network)
perhertz = slope * second
perspikeandsyn = perhertz / (1.0 * Hz * fixed_syn_spiking_network * N_spiking_network * defaultclock.dt)
subplot(335)
plot(rates_spiking_network, time_spiking_network, '.')
plot([0, max(rates_spiking_network)], [intercept + float(i) * slope for i in [0, max(rates_spiking_network)]])
title('Spiking')
xlabel('rate')

time_spiking_network_syn = [ do_time(spiking_network, rate_syn, syn) for syn in syn_spiking_network ]
slope, intercept, r, tt, stderr = scipy.stats.linregress(syn_spiking_network, time_spiking_network_syn)
persyn = slope * second / (rate_syn * N_spiking_network * defaultclock.dt)
perspike = (perspikeandsyn - persyn) * fixed_syn_spiking_network
print 'Time per synapse:', persyn
print 'Time per spike:', perspike
subplot(336)
plot(syn_spiking_network, time_spiking_network_syn, '.')
plot([0, max(syn_spiking_network)], [intercept + float(i) * slope for i in [0, max(syn_spiking_network)]])
title('Spiking')
xlabel('synapses per neuron')

time_threshold = [ do_time(threshold, N) for N in N_threshold ]
slope, intercept, r, tt, stderr = scipy.stats.linregress(N_threshold, time_threshold)
threshold_perneuron = slope * second
print 'Thresholding operation per neuron:', threshold_perneuron
subplot(338)
plot(N_threshold, time_threshold, '.')
plot([0, max(N_threshold)], [intercept + float(i) * slope for i in [0, max(N_threshold)]])
title('Thresholding')

time_threshold_cachemiss = [ do_time(threshold_cachemiss, N) for N in N_threshold ]
slope, intercept, r, tt, stderr = scipy.stats.linregress(N_threshold, time_threshold_cachemiss)
threshold_perneuron_cachemiss = slope * second
print 'Thresholding operation per neuron, cachemiss:', threshold_perneuron_cachemiss
subplot(339)
plot(N_threshold, time_threshold_cachemiss, '.')
plot([0, max(N_threshold)], [intercept + float(i) * slope for i in [0, max(N_threshold)]])
title('Thresholding cachemiss')

show()
