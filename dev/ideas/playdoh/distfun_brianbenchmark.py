from brian import NeuronGroup, SpikeCounter, Network, ms, second, isscalar

def fun(sigma, args):
    """
    This function computes the mean firing rate of a LIF neuron with
    white noise input current (OU process with threshold).
    """
    if not isscalar(sigma):
        raise Exception('sigma must be a scalar')
    N = args['N']
    tau = args['tau']
    model = args['model']
    reset = args['reset']
    threshold = args['threshold']
    duration = args['duration']
    G = NeuronGroup(N, model=model, reset=reset, threshold=threshold)
    M = SpikeCounter(G)
    net = Network(G, M)
    net.run(duration)
    r = M.nspikes * 1.0 / N
    return r

if __name__ == '__main__':
    from playdoh import *
    import numpy as np
    import time

    N = 2000
    tau = 10 * ms
    model = '''dV/dt=-V/tau+sigma*(2/tau)**.5*xi : 1'''
    reset = 0
    threshold = 1
    duration = 1 * second

    sigmas = list(np.linspace(.3, .6, 8))

    t = time.clock()
    for s in sigmas:
        r = fun(s, dict(N=N,
                           tau=tau,
                           model=model,
                           reset=reset,
                           threshold=threshold,
                           duration=duration))
        print "Mean firing rate with sigma = %.2f : %.2f Hz" % (s, r)
    print "Serial time = %.3f s" % (time.clock() - t)
    print

    dfun = distribute(fun, dict(N=N,
                                   tau=tau,
                                   model=model,
                                   reset=reset,
                                   threshold=threshold,
                                   duration=duration))

    t = time.clock()
    results = dfun(sigmas)
    t = time.clock() - t

    for s, r in zip(sigmas, results):
        print "Mean firing rate with sigma = %.2f : %.2f Hz" % (s, r)
    print "Parallel time = %.3f s" % t

