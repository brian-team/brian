from brian import NeuronGroup, SpikeCounter, Network, ms, second, isscalar

"""
This script shows how to parallelize a Brian script without effort.

Define a function as follows :

def fun(x, **args):
    # ...
    return r

This function defines a NeuronGroup and does some operations before returning
a result r.
'x' is a parameter, and you want to calculate r for different values of x.
Also, there are several static parameters in **args that do not change.

If you call
r = fun(x, arg1=val1, ...)

then your function should return the right result. If you want to evaluate fun
with several values of x, then you would do :

xlist = ...
rlist = zeros(len(xlist))
for i in xrange(len(xlist)):
    rlist[i] = fun(xlist[i], arg1=val1, ...) 
    
This code snippet can be accelerated if you have multiple CPUs by doing this :

import brian.dev.ideas.distributedfunction.distributedfunction as df
dfun = df.DistributedFunction(fun, arg1=val1, ...)
xlist = ...
rlist = dfun(xlist)

That's it!
"""

def fun(sigma, **args):
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
    G = NeuronGroup(N, model = model, reset = reset, threshold = threshold)
    M = SpikeCounter(G)
    net = Network(G, M)
    net.run(duration)
    r = M.nspikes*1.0/N
    return r

if __name__ == '__main__':
    import distributedfunction as df
    import numpy as np
    import time
    
    N = 2000
    tau = 10*ms
    model = '''dV/dt=-V/tau+sigma*(2/tau)**.5*xi : 1'''
    reset = 0
    threshold = 1
    duration = 1*second
    
    sigmas = list(np.linspace(.3, .6, 8))
    
    t = time.clock()
    for s in sigmas:
        r = fun(s, N=N,
                   tau=tau,
                   model=model,
                   reset=reset,
                   threshold=threshold,
                   duration=duration)
        print "Mean firing rate with sigma = %.2f : %.2f Hz" % (s, r)
    print "Serial time = %.3f s" % (time.clock()-t)
    print
    
    dfun = df.DistributedFunction(fun, N=N,
                                       tau=tau,
                                       model=model,
                                       reset=reset,
                                       threshold=threshold,
                                       duration=duration)
    
    t = time.clock()
    results = dfun(sigmas)
    t = time.clock()-t
    
    for s,r in zip(sigmas, results):
        print "Mean firing rate with sigma = %.2f : %.2f Hz" % (s, r)
    print "Parallel time = %.3f s" % t
    
    