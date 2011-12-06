speedtrack_desc = 'MultiSpikeGeneratorGroup with a fixed array of spiketimes'

from brian import *
from tracking import *
import gc, time

# All neurons fire simultaneously with the given rate
# The spiketimes are either given as a list of (index, time) pairs or as an 
# array with first column index and second column time  

# Firing rates
rates = [1, 500, 1000, 2000, 5000] #Hz

def run_barrage():
    results = []    
    for r in rates:                    
        print 'Rate: %dHz ' % (r)
        t = TrackSpeed(r)
        results.append((str(r), track(t), r))
        gc.collect()
    return (speedtrack_desc, results)

def plot_results(name, results):
    _, t, arg = zip(*results)
    plot(arg, t, 'o-')
    xlabel('rate (Hz)')
    ylabel('time (s)')
    title(name)


class TrackSpeed(object):
             
    def __init__(self, rate, duration=1): 
        
        N = 100
        dt = defaultclock.dt
        defaultclock.reinit()
        self.duration = duration
        
        #The spike times are shifted by half a dt to center the spikes in the bins
        times = arange(0, duration, 1./rate)             
        spiketimes = [[t*second + dt/2 for t in times] for n in xrange(N)]          
        
        start = time.time()
        G = MultipleSpikeGeneratorGroup(spiketimes)
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
        print desc, args, ':', t
    plot_results(name, results)
    show()
