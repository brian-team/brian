import time
from brian import *
from coincidence_counter import *
from vectorized_neurongroup import *
from vectorized_monitor import *
#from nose.tools import *


def test_basic():
    delta = 2*ms
    
#    target_train = [10*ms, 17.999999*ms, 30*ms]
#    source_train = [10*ms, 20*ms, 30*ms]
#    
#    target = [(0,t) for t in target_train]
#    source = [(0,t) for t in source_train]


    data = [(1, 9.6 * ms), (0, 9.7 * ms), (1, 19.3 * ms), (0, 19.5 * ms), (1, 29.0 * ms), (0, 29.3 * ms), (1, 38.7 * ms), (0, 39.1 * ms), (1, 48.4 * ms), (0, 48.9 * ms), (1, 58.1 * ms), (0, 58.7 * ms), (1, 67.8 * ms), (0, 68.5 * ms), (1, 77.5 * ms), (0, 78.3 * ms), (1, 87.2 * ms), (0, 88.1 * ms), (1, 96.9 * ms), (0, 97.9 * ms), (1, 0.1066* second), (0, 0.1077* second), (1, 0.1163* second), (0, 0.1175* second), (1, 0.126* second), (0, 0.1273* second), (1, 0.1357* second), (0, 0.1371* second), (1, 0.1454* second), (0, 0.1469* second), (1, 0.1551* second), (0, 0.1567* second), (1, 0.1648* second), (0, 0.1665* second), (1, 0.1745* second), (0, 0.1763* second), (1, 0.1842* second), (0, 0.1861* second), (1, 0.1939* second), (0, 0.1959* second)]
#    data = [ (0, 162.5*ms), (1, 164.1 *ms), (0,172.3*ms), (1, 174.3*ms)]
    
    source = [(i,t) for i,t in data if i == 0]
    target = [(i,t) for i,t in data if i == 1]
    
    source_train = [t for i,t in source]
    target_train = [t for i,t in target]

    duration = maximum(target_train[-1], source_train[-1])*second
    
    group = SpikeGeneratorGroup(N = 1, spiketimes = source)
    cd = CoincidenceCounter(source = group, data = target_train, delta = delta)
    run(duration)
    
    online = cd.coincidences[0]
    offline = gamma_factor(source = source_train, target = target_train, delta = delta)
    
    print "online =", online
    print "offline =", offline
    

def test_if():
    """
    Simulates an IF model with constant input current and checks
    the total number of coincidences with prediction.
    """
    eqs = """
    dV/dt = -V/tau+I : 1
    tau : second
    I : Hz
    """ 
    N = 2
    taus = [30*ms, 31*ms]
    duration = 400*ms
    input = 120.0/second * ones(int(duration/defaultclock._dt))
    delta = 4*ms

    # Generates data from an IF neuron
    group = NeuronGroup(N = N, model = eqs, reset = 0, threshold = 1)
    group.tau = taus
    group.I = TimedArray(input)
    M = SpikeMonitor(group)

    run(duration)
    data = M.spikes
#    print data
    
    reinit_default_clock()
    
    group = NeuronGroup(N = 1, model = eqs, reset = 0, threshold = 1)
    group.tau = taus[0]
    group.I = TimedArray(input)
    
    train0 = [t for i,t in data if i == 0]
    train1 = [t for i,t in data if i == 1]
    cd = CoincidenceCounter(source = group, data = train1, delta = delta)
    run(duration)
    
    online_gamma = cd.gamma[0]
    offline_gamma = gamma_factor(train0, train1, delta = delta)
    
    print "online gamma =", online_gamma
    print "offline gamma =", offline_gamma


test_if()
