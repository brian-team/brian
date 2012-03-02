from brian import *
from brian.experimental.modelfitting import *
from numpy.random import *
import time

if __name__ == '__main__':
    
    equations = Equations('''
        dV/dt=(R*I-V)/tau : 1
        I : 1
        R : 1
        tau : second
    ''')
    
    input = loadtxt('current.txt')
    trace = loadtxt('trace_artificial.txt')
    
    groups = 1
    overlap = 0*ms
#    input, trace = slice_trace(input, trace, slices = groups, overlap = overlap)
    
    
    neurons = 25600
    
    R = 3e9*ones(neurons)
    tau = 25*ms*ones(neurons)
    
    criterion = LpError(p=2, varname='V')
    
    t0 = time.clock()
    criterion_values = simulate( model = equations,
                                reset = 0,
                                threshold = 1,
                                data = trace,
                                input = input,
                                use_gpu = True,
                                groups = groups,
                                overlap = overlap,
                                stepsize = 128*ms,
                                dt = .1*ms,
                                criterion = criterion,
                                neurons = neurons,
                                R = R,
                                tau = tau)
    dur = time.clock()-t0
    print dur
    print min(criterion_values), max(criterion_values) 
