from vbench.benchmark import Benchmark

setup = """
from brian import *
log_level_error() # do not show warnings    

N = 100
rate = 250
dt = defaultclock.dt
defaultclock.reinit()

#The spike times are shifted by half a dt to center the spikes in the bins
times = arange(0, 1., 1. / rate)                         
spiketimes = [[t*second + dt/2 for t in times] for n in xrange(N)]                    
"""

#to do a fair comparison between old and new versions, this includes the setting
#up of the SpikeGeneratorGroup, because the newer version does quite some
#preprocessing of the data at this point.
statement = '''
G = MultipleSpikeGeneratorGroup(spiketimes)
net = Network(G)
net.prepare()
net.run(1 * second)
'''

bench_multiple = Benchmark(statement, setup,
                         name='MultipleSpikeGeneratorGroup with list of lists')
