"""
Loads an AER .dat file and plays it in Brian

This works (more or less).

The main bottleneck is creating the SpikeGeneratorGroup,
specifically the gather() method.
"""
from brian import *
from brian.experimental.neuromorphic import *
from time import time

#defaultclock.dt=1*ms

path=r'C:\Users\Romain\Desktop\jaerSampleData\DVS128'
filename=r'\Tmpdiff128-2006-02-03T14-39-45-0800-0 tobi eye.dat'

addr,timestamp=load_AER(path+filename)
print len(addr),"events"
#print "Timestamps sorted:",all(diff(timestamp)>=0) # Check if sorted
x,y,pol=extract_DVS_event(addr)
spiketimes=array((y,timestamp*1e-6)).T

t0=time()
#P=SpikeGeneratorGroup(128,spiketimes,gather=False)
# True: 3.66 (3.55 for the bottleneck) + 1.45
# False: 0.04 + 0.75 (Running is also faster!!)
P=SpikeQueue(128,timestamp*1e-6,y,check_sorted=False)
# 0.88 + .47 (Running is faster, but construction is slower)
t00=time()
print "Group creation:",t00-t0
M=SpikeMonitor(P)

t0=time()
run(1*second)
t00=time()
print "Running time:",t00-t0

# Checking
spikes=array(zip(*M.spikes)[1])
print timestamp[:10],spikes[:10]*1e6

raster_plot(M)
show()
