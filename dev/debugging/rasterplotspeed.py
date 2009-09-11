from brian import *
from time import time

N = 1000
duration = 1*second
nspikes = int(N*duration*50*Hz)

G = NeuronGroup(N, 'V:1')
M = SpikeMonitor(G)
M2 = SpikeMonitor(G)

M.spikes = [(randint(N), rand()*duration) for _ in xrange(nspikes)]
M.spikes.sort(key=lambda (a,b):b)
M2.spikes = M.spikes

start = time()
raster_plot(M)
print time()-start
#show()