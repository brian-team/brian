import time
t1 = time.clock()
from brian import *
t2 = time.clock()-t1
print t2
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
    NTarget = 1
    group_size = 10
    N = NTarget * group_size
    tau = .04+.02*rand(N)
    dt = .1*ms
    duration = 400*ms
    I = 120.0/second + 5.0/second * randn(int(duration/dt))

    # Generates data from an IF neuron
    vgroup = VectorizedNeuronGroup(model = eqs, reset = 0, threshold = 1, 
             input_name = 'I', input_values = I, dt = dt, 
             tau = tau)
    M = SpikeMonitor(vgroup)
    net = Network(vgroup, M)
    net.run(duration)
    data = M.spikes
    
    # Runs simulation
    vgroup = VectorizedNeuronGroup(model = eqs, reset = 0, threshold = 1,
                        input_name = 'I', input_values = I,
                        dt = dt, 
                        tau = tau)
    model_target = kron(arange(NTarget), 10)
    cd = CoincidenceCounter(vgroup, data, model_target = model_target, delta = .005)
    M = VectorizedSpikeMonitor(group)
    
    net = Network(vgroup, cd)
    reinit_default_clock()
    cd.reinit()
    
    run(group.duration)

    gamma = cd.gamma

test()
