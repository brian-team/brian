#!/usr/bin/env python
'''
Example of using Python multiprocessing module to distribute simulations over
multiple processors.

The general procedure for using multiprocessing is to define and run a network
inside a function, and then use multiprocessing.Pool.map to call the function
with multiple parameter values. Note that on Windows, any code that should only
run once should be placed inside an if __name__=='__main__' block.
'''

from brian import *
import multiprocessing

# This is the function that we want to compute for various different parameters
def how_many_spikes(excitatory_weight):
    # These two lines reset the clock to 0 and clear any remaining data so that
    # memory use doesn't build up over multiple runs.
    reinit_default_clock()
    clear(True)
    eqs = '''
    dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
    '''
    P = NeuronGroup(4000, eqs, threshold= -50 * mV, reset= -60 * mV)
    P.v = -60 * mV + 10 * mV * rand(len(P))
    Pe = P.subgroup(3200)
    Pi = P.subgroup(800)
    Ce = Connection(Pe, P, 'ge')
    Ci = Connection(Pi, P, 'gi')
    Ce.connect_random(Pe, P, 0.02, weight=excitatory_weight)
    Ci.connect_random(Pi, P, 0.02, weight= -9 * mV)
    M = SpikeMonitor(P)
    run(100 * ms)
    return M.nspikes

if __name__ == '__main__':
    # Note that on Windows platforms, all code that is executed rather than
    # just defining functions and classes has to be in the if __name__=='__main__'
    # block, otherwise it will be executed by each process that starts. This
    # isn't a problem on Linux.
    pool = multiprocessing.Pool() # uses num_cpu processes by default
    weights = linspace(0, 3.5, 100) * mV
    args = [w * volt for w in weights]
    results = pool.map(how_many_spikes, args) # launches multiple processes
    plot(weights, results, '.')
    show()
