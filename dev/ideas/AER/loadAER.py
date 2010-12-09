"""
Loads an AER .dat file and plays it in Brian

Version:
1 -    addr is 2 bytes, timestamp is 4 bytes (default)
2 -    addr is 4 bytes, timestamp is 4 bytes
"""
from brian import *
from brian.experimental.neuromorphic import *

path=r'C:\Users\Romain\Desktop\jaerSampleData\DVS128'
filename=r'\Tmpdiff128-2006-02-03T14-39-45-0800-0 tobi eye.dat'

addr,timestamp=load_AER(path+filename)
dt=defaultclock.dt
#multiplier=int(dt/(1e-6*second)) # number of ticks per dt
timestamp_dt=array(timestamp*1e-6/dt,dtype=int) # in units of dt
u,indices=unique(timestamp_dt,return_index=True) # split over timesteps
#=SpikeContainer(100) # number of bins
#spikes=[(pixel_to_neuron(*extract_DVS_event(ad)),t*1e-6*second) for (ad,t) in zip(addr,timestamp)]
x,y,pol=extract_DVS_event(addr)
spikes=[(yy,t*1e-6*second) for (yy,t) in zip(y,timestamp)]
# assuming microsecs
print spikes[-1]

P=SpikeGeneratorGroup(128,spikes)
M=SpikeMonitor(P)

print "starting"
run(1*second)

raster_plot(M)
show()
