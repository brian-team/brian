'''
Model fitting example.
Fits an integrate-and-fire model to an in-vitro electrophysiological 
recording during one second.
'''
from brian import loadtxt, ms, Equations
from brian.library.modelfitting import *

if __name__ == '__main__':
    equations = Equations('''
        dV/dt=(R*I-V)/tau : 1
        I : 1
        R : 1
        tau : second
    ''')
    input = loadtxt('current.txt')
    spikes = loadtxt('spikes.txt')
    
    results = modelfitting(model=equations, reset=0, threshold=1,
                                 data=spikes,
                                 input=input, dt=.1 * ms,
                                 popsize=1000, maxiter=1, delta=4 * ms,
                                 cpu=1,
#                                 scheme=rk2_scheme, # can use euler_scheme, exp_euler_scheme (for HH), or rk2_scheme
                                 R_initrange=[1.0e9, 9.0e9],
                                 tau_initrange=[10 * ms, 40 * ms])

    print_table(results)
