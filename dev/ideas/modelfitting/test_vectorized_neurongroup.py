from brian import *
from vectorized_neurongroup import *

def test():
    model = """
    dV/dt = -V/tau+I : 1
    tau : second
    I : Hz
    """
    reset = 0
    threshold = 1
    duration = 400*ms
    overlap = 50*ms
    input = arange(50.0, 100.0, 50.0/int(duration/defaultclock.dt))
    tau = arange(.01,.07,.001)
    N = len(tau)
    slices = 4
    vgroup = VectorizedNeuronGroup(model = model, reset = reset, threshold = threshold, 
                                   input = input, overlap = overlap, 
                                   slices = slices, tau = tau)
    M = SpikeMonitor(vgroup)
    net = Network(vgroup, M)
    net.run(vgroup.duration)
    
    # TODO: for now, just checks that the code runs !
    assert True


if __name__ == '__main__':
    test()
    