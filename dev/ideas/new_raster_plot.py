'''
A different way of plotting raster plots
'''
from brian import *

tau=5*ms
N=10
f=50*Hz
a=.2

eqs='''
dv/dt=(v0-v+a*(sin(2*pi*f*t)+sin(4*pi*f*t)))/tau : 1
v0 : 1
'''

neurons=NeuronGroup(N,model=eqs,threshold=1,reset=0)
neurons.v0=1.37
neurons.v=rand(N)

run(1*second)
S=SpikeMonitor(neurons)
run(100*ms)
#raster_plot(S)
for i,t in S.spikes:
    plot([t,t],[i,i+.9],'b',linewidth=2)
show()
