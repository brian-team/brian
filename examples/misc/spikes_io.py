#!/usr/bin/env python
'''
This script demonstrates how to save/load spikes in AER format from inside Brian.
'''
from brian import *

####################### SAVING #########################

# First we need to generate some spikes
N = 1000
g = PoissonGroup(N, 200*Hz)

# And set up a monitor to record those spikes to the disk
Maer = AERSpikeMonitor(g, './dummy.aedat')

# Now we can run
run(100*ms)

# This line executed automatically when the script ends, but here we
# need to close the file because we re-use it from within the same script
Maer.close()


clear(all = True)
reinit_default_clock()
####################### LOADING ########################

# Now we can re-load the spikes
addr, timestamps = load_aer('./dummy.aedat')

# Feed them to a SpikeGeneratorGroup
group = SpikeGeneratorGroup(N, (addr, timestamps))

# The group can now be used as any other, here we choose to monitor
# the spikes
newM = SpikeMonitor(group, record = True)

run(100*ms)

# And plot the result
raster_plot(newM)
show()


