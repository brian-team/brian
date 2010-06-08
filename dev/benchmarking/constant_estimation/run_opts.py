from brian import *

duration = 1 * second
defaultclock.dt = 0.1 * ms

max_groups = 5
max_connections = 5
N_linear_network = [100, 200, 300, 1000, 2000, 4000]
cachemiss_array_size = 300000 # should take up 2MB
N_multivar_linear_network = 1000
vars_linear_network = range(1, 6)
N_spiking_network = 1000
rates_spiking_network = [x * Hz for x in randint(20, 300, size=10)]
fixed_syn_spiking_network = 500
syn_spiking_network = list(randint(100, 1000, size=10))#+list(randint(1000,5000,size=10))
rate_syn = 50 * Hz
N_threshold = randint(100, 1000, size=20)
