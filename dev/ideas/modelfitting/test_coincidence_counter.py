import time
from brian import *
from coincidence_counter import *
from vectorized_neurongroup import *
from vectorized_monitor import *
from nose.tools import *

def test():
    """
    Simulates an IF model with constant input current and checks
    the total number of coincidences with prediction.
    """
    eqs = """
    dV/dt = -V/tau+I : 1
    tau : second
    I : Hz
    """
#    source = [10*ms, 20*ms]
#    target = [10*ms, 22.01*ms]
#    delta = 2*ms
#    gamma = gamma_factor(source, target, delta)
#    print gamma
#    exit()
    
    N = 2
    taus = [30*ms, 32*ms]
    duration = 200*ms
    input = 120.0/second * ones(int(duration/defaultclock._dt))
    delta = 2*ms

    # Generates data from an IF neuron
    group = NeuronGroup(N = N, model = eqs, reset = 0, threshold = 1)
    group.tau = taus
    group.I = TimedArray(input)
    M = SpikeMonitor(group)

    run(duration)
    data = M.spikes
    
    reinit_default_clock()
    
    group = NeuronGroup(N = 1, model = eqs, reset = 0, threshold = 1)
    group.tau = taus[0]
    group.I = TimedArray(input)
    
    train0 = [t for i,t in data if i == 0]
    train1 = [t for i,t in data if i == 1]
    cd = CoincidenceCounter(source = group, data = train1, delta = delta)
    run(duration)
    
    online_gamma = cd.gamma[0]
    online_coinc = cd.coincidences[0]
    offline_gamma = gamma_factor(train0, train1, delta = delta)
    offline_coinc = offline_gamma[0]
    offline_gamma = offline_gamma[1]
    
    print "online coinc =", online_coinc
    print "offline coinc =", offline_coinc
    print
    print "online gamma =", online_gamma
    print "offline gamma =", offline_gamma
    

test()
