# Top secret!
# Seems to go in the opposite direction!
from brian import *

defaultclock.dt=.01*ms

N=30000
tau=.5*ms
taue=.5*ms
sigma=.8
eqs='''
dv/dt=(ge-v)/tau : 1
dge/dt=-ge/taue + sigma/taue**.5*xi : 1
'''

T=5*ms # 200 Hz
p=10
maxITD=1*ms
w=.9
spiketimes=[]
for i in range(p):
    spiketimes+=[i*T,i*(T+maxITD/p)]
I=SpikeGeneratorGroup(1,[(0,t) for t in spiketimes])
print "Starting..."
P=NeuronGroup(N,model=eqs,threshold=1,reset=0,refractory=2*ms)
C=Connection(I,P,'ge')
C.connect_full(weight=w)

S=SpikeMonitor(P)
PSTH=PopulationRateMonitor(P)

run((p+1)*T)
subplot(211)
raster_plot(S)
subplot(212)
stimulus=zeros(len(PSTH.times))
for i in range(p):
    stimulus[int(i*(T+maxITD/p)/defaultclock.dt)]=250*Hz
plot((PSTH.times/ms) % 5.,PSTH.smooth_rate(.03*ms,'gaussian'))
plot((PSTH.times/ms) % 5.,stimulus)
show()
