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

    I = 120/second
    tau = arange(.03, .06, .01)
    N = len(tau)
    dt = .1*ms
    isi = -tau*log(1-1/(tau*I))
    duration = 120*ms
    
    data = []
    for i in range(N):
        for j in range(1,int(duration/isi[i])+1):
            t = int(j*isi[i]/dt)*dt
            if t <= duration:
                data += [(i,t)]
    data.sort(cmp=lambda x,y:2*int(x[1]>y[1])-1)

    group = VectorizedNeuronGroup(model = eqs, reset = 0, threshold = 1,
                        input_name = 'I', input_values = I*ones(int(duration/dt)),
                        dt = dt, overlap = 30*ms,slice_number = 1,
                        tau = tau)
    
    cd = CoincidenceCounter(group, data, model_target = arange(N), delta = .004)
    M = VectorizedSpikeMonitor(group)
    run(group.duration)

    assert cd.gamma.min() > .999999

test()
