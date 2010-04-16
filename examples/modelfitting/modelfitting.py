'''
Model fitting example.
Fits an integrate-and-fire model to an in-vitro electrophysiological 
recording during one second.
'''
from brian import *
from brian.library.modelfitting import *

if __name__ == '__main__':
    
    equations = '''
        dV/dt=(R*I-V)/tau : 1
        I : 1
        R : 1
        tau : second
    '''
    
    input = loadtxt('current.txt')
    spikes = loadtxt('spikes.txt')
    
    results = modelfitting(model = equations, reset = 0, threshold = 1, 
#                                 refractory = 5*ms,
                                 data = spikes, 
                                 input = input, dt = .1*ms,
                                 particles = 1000, iterations = 3, delta = 2*ms,
                                 max_cpu = 1,
                                 R = [1.0e9, 9.0e9], tau = [10*ms, 40*ms])
    
    print_results(results)
