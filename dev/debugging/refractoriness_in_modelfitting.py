from brian import *
from brian.library.modelfitting import *

if __name__ == '__main__':
    
    duration = 1*second

    model = Equations('''
        dV/dt=(R*I-V)/tau : 1
        I : 1
        R : 1
        tau : second
    ''')

    input = randn(int(duration/defaultclock.dt))*2e-10+4e-10
    
    def getspikes():
        G = NeuronGroup(1, model, reset=0, threshold=1, refractory=5*ms)
        G.R = 3e9
        G.tau = 20*ms
        M = SpikeMonitor(G)
        G.I = TimedArray(input)
        run(duration)
        i, t = zip(*M.spikes)
        clear(True)
        reinit_default_clock()
        return t
    
    spikes0 = getspikes()
    spikes = []
    for i in xrange(1):
        spikes.extend([(i, spike + 5*i*ms) for spike in spikes0])

    results = modelfitting( model = model,
                            reset = 0,
                            threshold = 1,
                            data = spikes,
                            input = input,
                            dt = .1*ms,
                            popsize = 5,
                            maxiter = 1,
                            gpu = 1,
                            #algorithm = CMAES,
                            delta = 1*ms,
                            R = [3.0e9, 3.0e9],
                            tau = [20*ms, 20*ms],
                            delays = [0*ms, 0*ms],
                            refractory=[5*ms, 5*ms],
                            )
#                            R = [1.0e9, 9.0e9],
#                            tau = [10*ms, 40*ms],
#                            delays = [-100*ms, 100*ms],
#                            refractory=[0*ms, 0*ms, 10*ms, 10*ms])
    print_table(results)
