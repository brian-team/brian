from brian import *
from coincidence_counter import *
from vectorized_neurongroup import *
from nose.tools import *


def test():
    """
    Simulates an IF model with constant input current and checks
    the total number of coincidences with prediction.
    """
    eqs = """
    dV/dt = -V/tau+I : 1
    tau : second
    I : Hz"""

    I = 120/second
    tau = 1/I/second + arange(.001, .005, .001)
    N = len(tau)
    n = 10
    dt = .1*ms
    isi = -tau*log(1-1/(tau*I))
    durations = n*isi
    duration = durations.min() 

    group = VectorizedNeuronGroup(
                        model = eqs,
                        reset = 0,
                        threshold = 1,
                        input_name = 'I',
                        input_values = I*ones(int(duration/dt)),
                        dt = dt, 
                        overlap = 0*ms,
                        slice_number = 1,
                        tau = tau)
    
    # we compute the predicted spike train
    data = []
    for i in range(N):
        data += [(i,floor(t/dt)*dt) for t in cumsum(isi[i]*ones(n)) if t <= duration]
    data.sort(cmp=lambda x,y:2*int(x[1]>y[1])-1)
    
    
    cd = CoincidenceCounter(group, data, model_target = arange(N))
    
    M = SpikeMonitor(group)
    run(group.duration)
    
    print int(duration/dt)*dt
    print cd.gamma
    print data
    print M.spikes

#    assert (cd.coincidences == n).all()

test()
