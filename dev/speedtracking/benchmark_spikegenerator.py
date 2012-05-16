from vbench.benchmark import Benchmark

common_setup = """
from brian import *
set_global_preferences(useweave=False, usecodegen=False, usecodegenweave=False)
log_level_error() # do not show warnings    
"""

setup_template = """
N = %(N)d
rate = 250
dt = defaultclock.dt
defaultclock.reinit()

#The spike times are shifted by half a dt to center the spikes in the bins
times = arange(0, 1., 1. / rate)             

stim_type = '%(stim_type)s' #will be replaced from outside 

if stim_type == 'pairs':            
    spiketimes = [(idx, t*second +dt/2) for idx in xrange(N) for t in times]            
elif stim_type == 'array':
    spiketimes = vstack([array([n, t]) for t in times for n in xrange(N)])                      
"""

# This tests the overhead of the SpikeGeneratorGroup if no spikes are generated
setup_empty = """
N = 1
spiketimes = []
"""

#to do a fair comparison between old and new versions, this includes the setting
#up of the SpikeGeneratorGroup, because the newer version does quite some
#preprocessing of the data at this point.
statement = '''
G = SpikeGeneratorGroup(N, spiketimes)
net = Network(G)
net.prepare()
net.run(1 * second)
'''

bench_pairs = Benchmark(statement,
                         common_setup + (setup_template % {'stim_type' : 'pairs',
                                                           'N' : 100}),
                         name='SpikeGeneratorGroup with (index, time) pairs')
bench_array = Benchmark(statement,
                          common_setup + (setup_template % {'stim_type' : 'array',
                                                            'N' : 200}),
                          name='SpikeGeneratorGroup with an array of indices/times')

bench_bigarray = Benchmark(statement,
                           common_setup + (setup_template % {'stim_type' : 'array',
                                                             'N' : 1000}),
                           name='SpikeGeneratorGroup with a big array of indices/times')

bench_empty = Benchmark(statement,
                        common_setup + setup_empty,
                        name='SpikeGeneratorGroup without any spikes')
