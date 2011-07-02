'''
This example demonstrates how to load AER data files and use them as a
SpikeGeneratorGroup in Brian


For more information either check the documentation in the package,
post to the google group, or send me an email victor.benichoux@ens.fr

'''
from brian import *
from brian.experimental.neuromorphic import *


filename = '/path/to/file' # support .aedat and .aeidx files

addr, timestamps = load_AER(filename) # load the data
# addr contains a list of addresses
# timestamps the list of spike times (the first spike comes in at t = 0, but you may change that
# Note: In the case of an aeidx file, the value returned is a list of tuples (addr, timestamp)

group = AERSpikeGeneratorGroup((addr, timestamps))# Create an AER group

# at that point you can do whatever you want, it is a regular Brian neuron group

# * Events addressing * 
#
# We provide with two event extraction functions that fetch the
# specialized addresses of the DVS retina and the AMS cochlea.
# For example:
(x, y, pol) = extract_DVS_event(addr) 
# returns the x and y positions of the neuron, alongside the polarity
# (on/off) of the event
# See the documentation for more on this.

M = SpikeMonitor(group)# monitor it

run(group.maxtime) # this group has an additional attribute maxtime
# that gives you the last spike time

# plot it
raster_plot(M)
show()

# Additionally you can save any SpikeMonitor to and aedat file format as follows
save(M, './dummy.aedat')
