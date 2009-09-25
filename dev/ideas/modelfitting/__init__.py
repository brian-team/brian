from brian import *
from coincidence_counter import *
from optimization import *
from modelfitting import *

if __name__ == '__main__':
    
    eqs = """
    dV/dt = -V/tau+I : 1
    tau : second
    I : Hz
    """
    NTarget = 1
    tau = .04+.02*rand(NTarget)
    dt = .1*ms
    duration = 400*ms
    I = 120.0/second + 5.0/second * randn(int(duration/dt))

    # Generates data from an IF neuron with tau between 20-40ms
    # TODO: Replace by TimedArray
    vgroup = VectorizedNeuronGroup(model = eqs, reset = 0, threshold = 1, 
             input_var = 'I', input = I,
             tau = tau)
    M = SpikeMonitor(vgroup)
    net = Network(vgroup, M)
    net.run(duration)
    data = M.spikes
    
    # Tries to find tau
    params, value = modelfitting(model = eqs, reset = 0, threshold = 1,
                               data = data,
                               input = I,
                               particles = 10,
                               iterations = 10,
                               tau = [1*ms, 20*ms, 40*ms, 100*ms],
                               delta = 5*ms#,
                               #init = dict(V=-60*mV,...)
                               )
    
    print "real tau =", tau
    print "computed tau =", params['tau']
    
    
    