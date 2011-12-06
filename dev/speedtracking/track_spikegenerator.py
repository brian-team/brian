speedtrack_desc = 'SpikeGeneratorGroup with a fixed array of spiketimes'

from brian import *
from tracking import *
import gc, time

# All neurons fire simultaneously with the given rate
# The spiketimes are either given as a list of (index, time) pairs or as an 
# array with first column index and second column time  

# Firing rates
rates = [1, 500, 1000, 2000, 5000] #Hz

# Spike times are given as pairs, as pairs but using gather=True, or as an array 
stim_types = ['pairs', 'array']

def run_barrage():
    results = []
    
    for stim_type in stim_types:
        for r in rates:
            print stim_type, str(r)
            t = TrackSpeed(r, stim_type)
            results.append(('%s (%dHz)' % (stim_type, r), track(t),
                            (r, stim_type)))
            gc.collect()
    return (speedtrack_desc, results)

def plot_results(name, results):
    for stim_type in stim_types:
        allt = [t for _, t, args in results  if args[1] == stim_type]
        alln = [args[0] for _, t, args in results if args[1] == stim_type]
        plot(alln, allt, 'o-', label=(stim_type))
    legend(frameon=False)
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
        gather = False
        
        if stim_type == 'pairs':            
            spiketimes = [(idx, t*second +dt/2) for idx in xrange(N) for t in times]            
        elif stim_type == 'array':
            spiketimes = vstack([array([n, t]) for t in times for n in xrange(N)])          
        else:
            raise ValueError('Unknown stim_type (should be "pairs" or "array)')            
        
        start = time.time()
        G = SpikeGeneratorGroup(N, spiketimes, gather=gather)
        self.init_time = time.time() - start

        self.net = Network(G)
        self.net.prepare()

    def run(self):
        #simulate the initialization time by waiting...        
        time.sleep(self.init_time)        
        
        self.net.run(self.duration * second)

if __name__ == '__main__':
    name, results = run_barrage()
    print name
    for desc, t, args in results:
        print desc, ':', t
    plot_results(name, results)
    show()
