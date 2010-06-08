from brian import NeuronGroup, SpikeCounter, Network, ms, second, isscalar

"""
This script shows how to distribute a Brian script without effort.

Define a function as follows :

def fun(x, args):
    # ...
    return r

This function defines a NeuronGroup and does some operations before returning
a result r.
'x' is a parameter, and you want to calculate r for different values of x.
Also, there are several static parameters in args that do not change.

If you call
r = fun(x, dict(arg1=val1, ...))

then your function should return the right result. If you want to evaluate fun
with several values of x, then you would do :

xlist = ...
rlist = zeros(len(xlist))
for i in xrange(len(xlist)):
    rlist[i] = fun(xlist[i], dict(arg1=val1, ...)) 
    
This code snippet can be accelerated if you have multiple CPUs by doing this :

from playdoh import *
dfun = distribute(fun, dict(arg1=val1, ...))
xlist = ...
rlist = dfun(xlist)

That's it!
"""

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

    N = 2000
    tau = 10 * ms
    model = '''dV/dt=-V/tau+sigma*(2/tau)**.5*xi : 1'''
    reset = 0
    threshold = 1
    duration = 1 * second
    sigmas = [.3, .4, .5, .6]

    dfun = distribute(fun, dict(N=N, tau=tau, duration=duration,
                                       model=model, reset=reset, threshold=threshold))

    rates = dfun(sigmas)
    print rates
