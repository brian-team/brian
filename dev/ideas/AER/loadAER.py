"""
Loads an AER .dat file and plays it in Brian

Version:
1 -    addr is 2 bytes, timestamp is 4 bytes (default)
2 -    addr is 4 bytes, timestamp is 4 bytes
"""
from brian import *
from brian.experimental.neuromorphic import *

def pixel_to_neuron(x,y,pol):
    return y+0*x # let's just have 128 neurons, one per row for now

path=r'C:\Users\Romain\Desktop\jaerSampleData\DVS128'
filename=r'\Tmpdiff128-2006-02-03T14-39-45-0800-0 tobi eye.dat'

events=load_AER(path+filename)
spikes=[(pixel_to_neuron(*extract_DVS_event(addr)),t*1e-6*second) for (addr,t) in events]
# assuming microsecs
print spikes[-1]

P=SpikeGeneratorGroup(128,spikes)
M=SpikeMonitor(P)

print "starting"
run(1*second)

raster_plot(M)
show()
