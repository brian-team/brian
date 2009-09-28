from modelfitting import *

if __name__ == '__main__':
    from brian import *
    eqs = """
    dV/dt = -V/tau+I : 1
    tau : second
    I : Hz
    """
    NTarget = 2
    taus = .03+.03*rand(NTarget)
    duration = 1000*ms
    input = 120.0/second + 0.0/second * randn(int(duration/defaultclock.dt))

    # Generates data from an IF neuron with tau between 20-40ms
    group = NeuronGroup(N = NTarget, model = eqs, reset = 0, threshold = 1)
    group.tau = taus
    group.I = TimedArray(input)
    M = SpikeMonitor(group)
    run(duration)
    data = M.spikes
    
    # TODO : initial values for V : init = dict(V=-60*mV,...)
    # Tries to find tau
    params, value = modelfitting(model = eqs, reset = 0, threshold = 1,
                               data = data,
                               input = input,
                               particles = 50,
                               iterations = 5,
                               slices = 4,
                               overlap = 100*ms,
                               tau = [-inf, 30*ms, 60*ms, +inf],
                               delta = .5*ms)
    
    for i in range(NTarget):
        real_tau = taus[i]*1000
        computed_tau = params['tau'][i]*1000
        error = 100.0*abs(real_tau-computed_tau)/real_tau
        print "%d. Real tau = %.2f ms, found tau = %.2f, rel. error = %.2f %%" % \
            (i+1, real_tau, computed_tau, error)
    