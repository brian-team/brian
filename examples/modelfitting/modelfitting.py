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
    optinfo = dict([])
    optinfo['Minterval'] = 10
    input = loadtxt('current.txt')
    spikes = loadtxt('spikes.txt')

    results = modelfitting(model=equations, reset=0, threshold=1,
                                 data=spikes,
                                 input=input, dt=.1 * ms,
                                 particles=1000, iterations=10, delta=2 * ms,
                                 use_gpu=True, max_cpu=4, max_gpu=1,
                                 scheme=rk2_scheme, # can use euler_scheme, exp_euler_scheme (for HH), or rk2_scheme
                                 R=[1.0e9, 9.0e9], tau=[10 * ms, 40 * ms], optalg=GA, optinfo=optinfo)

    print_results(results)
