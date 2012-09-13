#!/usr/bin/env python
'''
Example showing how to fit a single model
with different target spike trains (several groups).
'''
from brian import loadtxt, ms, Equations, second
from brian.library.modelfitting import *

if __name__ == '__main__':

    model = Equations('''
        dV/dt=(R*I-V)/tau : 1
        I : 1
        R : 1
        tau : second
    ''')

    input = loadtxt('current.txt')
    spikes0 = loadtxt('spikes.txt')
    spikes = []
    for i in xrange(2):
        spikes.extend([(i, spike*second + 5*i*ms) for spike in spikes0])

    results = modelfitting( model = model,
                            reset = 0,
                            threshold = 1,
                            data = spikes,
                            input = input,
                            dt = .1*ms,
                            popsize = 1000,
                            maxiter = 3,
                            cpu = 1,
                            delta = 4*ms,
                            R = [1.0e9, 9.0e9],
                            tau = [10*ms, 40*ms],
                            delays = [-10*ms, 10*ms])
    print_table(results)

