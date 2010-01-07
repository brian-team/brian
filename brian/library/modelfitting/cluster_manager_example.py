from cluster_modelfitting import *

if __name__=='__main__':
    equations = Equations('''
        dV/dt=(R*I-V)/tau : 1
        I : 1
        R : 1
        tau : second
    ''')
    
    input = loadtxt('current.txt')[:10000]
    spikes = loadtxt('spikes.txt')
    print len(spikes)
    
    machines = [
                'Cyrille-Ulm',
                #'Astrance',
                ]
    
    params, gamma = modelfitting(model = equations, reset = 0, threshold = 1, 
                                 data = spikes, 
                                 input = input, dt = .1*ms,
                                 use_gpu = False, max_cpu = None, max_gpu = None,
                                 machines = machines,
                                 named_pipe = True,
                                 particles = 800, iterations = 1, delta = 2*ms,
                                 R = [1.0e9, 1.0e10], tau = [1*ms, 50*ms])
    
    print params
