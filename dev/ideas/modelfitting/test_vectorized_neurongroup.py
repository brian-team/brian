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
    dt = defaultclock.dt
    I = arange(50.0, 100.0, 50.0/int(duration/dt))
    tau = arange(.01,.07,.001)
    N = len(tau)
    slice_number = 4
    vgroup = VectorizedNeuronGroup(model = model, reset = reset, threshold = threshold, 
                                   input_name = 'I', input_values = I, overlap = overlap, 
                                   slice_number = slice_number, tau = tau)
    M = SpikeMonitor(vgroup)
    net = Network(vgroup, M)
    net.run(vgroup.duration)
    
    # TODO: alter SpikeMonitor to support vectorizedneurongroup
    M.spikes = [(mod(i,N),t+i/N*(vgroup.duration-overlap)*second) for i,t in M.spikes if t >= overlap]
    # TODO: for now, just checks that the code runs !
    assert True


if __name__ == '__main__':
    test()
    