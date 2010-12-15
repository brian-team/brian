"""
Loads an AER .dat file and plays it in Brian

Version:
1 -    addr is 2 bytes, timestamp is 4 bytes (default)
2 -    addr is 4 bytes, timestamp is 4 bytes
"""
from brian import *
from brian.experimental.neuromorphic import *

#defaultclock.dt=1*ms

path=r'C:\Users\Romain\Desktop\jaerSampleData\DVS128'
filename=r'\Tmpdiff128-2006-02-03T14-39-45-0800-0 tobi eye.dat'

addr,timestamp=load_AER(path+filename)
print len(addr),"events"
x,y,pol=extract_DVS_event(addr)
spiketimes=array((y,timestamp*1e-6)).T

P=SpikeGeneratorGroup(128,spiketimes,gather=True)
M=SpikeMonitor(P)

run(1*second)

raster_plot(M)
show()
