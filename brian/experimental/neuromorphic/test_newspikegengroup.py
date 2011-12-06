from brian import *
from brian.experimental.neuromorphic import *

defaultclock.dt = 1*ms
runtime = 100*ms

addr, timestamps = load_AER('./dummy.aedat')

N = max(addr)
group = NewSpikeGeneratorGroup(N, (addr, timestamps), time_unit = 1*usecond)

M = SpikeMonitor(group)

run(100*ms)

raster_plot(M)
show()
