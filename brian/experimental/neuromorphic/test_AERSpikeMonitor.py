from brian import *
from brian.experimental.neuromorphic import *

defaultclock.dt = 1*ms
runtime = 100*ms
g = PoissonGroup(100, 100*Hz)
#g = SpikeGeneratorGroup(1, [(0,t*ms) for t in range(int(runtime/defaultclock.dt))])

Maer = AERSpikeMonitor(g, './dummy.aedat')
M = SpikeMonitor(g)

run(100*ms)

Maer.close_file()


addr, timestamps = load_AER('./dummy.aedat')
if len(addr) == len(M.spikes):
    print 'looks good'
else:
    print 'glub'
    print 'addr, M.spikes',len(addr),len(M.spikes)
plot(timestamps, addr, '.')
show()
