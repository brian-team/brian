speedtrack_desc = 'SpikeGeneratorGroup with a fixed array of spiketimes'

from brian import *
from tracking import *
import gc

#Two conditions are tested:
#    * all neurons spiking with a certain density
#    * only one neuron (drawn randomly for every spike) spikes

# Firing rates
rates = [1, 1000, 2500, 5000, 10000] #Hz

# dt is also varied but should not make a difference
dts = [0.1, 0.01] #ms

# All neurons fire vs. one neuron fires
stim_types = ['all', 'random']

def run_barrage():
    results = []
    
    for stim_type in stim_types:
        for r in rates:
            for dt in dts:        
                t = TrackSpeed(r, stim_type)
                results.append((str(r), track(t), (r, stim_type, dt)))
                gc.collect()
    return (speedtrack_desc, results)

def plot_results(name, results):
    for stim_type in stim_types:
        for dt in dts:
            allt = [t for _, t, args in results 
                    if args[1] == stim_type and args[2] == dt]
            alln = [args[0] for _, t, args in results 
                    if args[1] == stim_type and args[2] == dt]
            plot(alln, allt, 'o-', label=('%s (dt=%.2fms)' % (stim_type, dt)))
    legend()
    xlabel('rate (Hz)')
    ylabel('time (s)')
    title(name)


class TrackSpeed(object):
             
    def __init__(self, rate, stim_type, duration=1): 
        
        N = 100
        dt = defaultclock.dt
        defaultclock.reinit()
        self.duration = duration
        
        #The spike times are shifted by half a dt to center the spikes in the bins
        times = arange(0, duration, 1./rate)             
        if stim_type == 'all':            
            spiketimes = [(idx, t*second +dt/2) for idx in xrange(N) for t in times]            
        elif stim_type == 'random':
            spiketimes = [(randint(N), t*second + dt/2) for t in times]
            pass
        else:
            raise ValueError('Unknown stim_type (should be "all" or "random')
        G = SpikeGeneratorGroup(N, spiketimes)

        self.net = Network(G)
        self.net.prepare()

    def run(self):
        self.net.run(self.duration * second)

if __name__ == '__main__':
    name, results = run_barrage()
    print name
    for desc, t, args in results:
        print desc, args, ':', t
    plot_results(name, results)
    show()
