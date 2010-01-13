from cluster_modelfitting_fast import *
#from cluster_modelfitting import *

if __name__=='__main__':
    equations = Equations('''
        dV/dt=(R*I-V)/tau : 1
        I : 1
        R : 1
        tau : second
    ''')
    
    input = loadtxt('current.txt')
    spikes = loadtxt('spikes.txt')
    
    machines = [
                #'Cyrille-Ulm',
                #'Romain-PC',
                #'Astrance',
                ]
    
    params, gamma = modelfitting(model = equations, reset = 0, threshold = 1, 
                                 data = spikes, 
                                 input = input, dt = .1*ms,
                                 use_gpu = False, max_cpu = 4, max_gpu = None,
                                 machines = machines,
                                 named_pipe = True,
                                 particles = 100000, iterations = 3, delta = 2*ms,
                                 R = [1.0e9, 1.0e9, 1.0e10, 1.0e10],
                                 tau = [1*ms, 1*ms, 50*ms, 50*ms])
    
    print params
