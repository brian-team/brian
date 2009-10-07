from modelfitting import *

if __name__ == '__main__':
    from brian import *
    eqs = Equations("""
    dV/dt = (-V+I)/tau : 1
    tau : second
    I : 1
    """)
    NTarget = 10
    #taus = .03+.03*rand(NTarget)
    taus = .01+.09*rand(NTarget)
    duration = 800*ms
    input = 3.0 * ones(int(duration/defaultclock.dt))

    # Generates data from an IF neuron with tau between 20-40ms
    group = NeuronGroup(N = NTarget, model = eqs, reset = 0, threshold = 1)
    group.tau = taus
    group.I = TimedArray(input)
    M = SpikeMonitor(group)
    run(duration)
    data = M.spikes
    
    # Tries to find tau
    params = modelfitting(model = eqs, reset = 0, threshold = 1,
                               data = data,
                               input = input,
                               particles = 20000,
                               iterations = 50,
                               slices = 1,
                               overlap = 250*ms,
                               tau = [10*ms, 10*ms, 100*ms, 100*ms],
                               delta = 1*ms)
    
    for i in range(NTarget):
        real_tau = taus[i]*1000
        computed_tau = params['tau'][i]*1000
        error = 100.0*abs(real_tau-computed_tau)/real_tau
        print "%d. Real tau = %.2f ms, found tau = %.2f ms, rel. error = %.2f %%" % \
            (i+1, real_tau, computed_tau, error)
    