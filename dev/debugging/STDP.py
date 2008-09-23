''' 
STDP model from Song, Miller and Abbott 2000
NOT WORKING - Don't know what's wrong!
'''
from brian import *
from brian.stdunits import *
from scipy import rand
from pylab import hist
import time

clk=Clock(dt=0.1*ms)

# Parameters
gmax=0.015
gin=0.05
Ap=0.005*gmax
Am=-1.05*Ap
taup=20*ms
taum=20*ms
Ne=1000
Ni=200
rate_ex=15*Hz
rate_in=10*Hz
taum=20*ms
Vrest=-70*mV
Eex=0*mV
Ein=-70*mV
Vt=-54*mV
Vr=-60*mV
taue=5*ms
taui=5*ms

# Model
dv=lambda v,ge,gi: (Vrest-v+ge*(Eex-v)+gi*(Ein-v))/taum
dge=lambda v,ge,gi: -ge/taue
dgi=lambda v,ge,gi: -gi/taui

input_ex=PoissonGroup(Ne,[rate_ex]*Ne,clk)
input_in=PoissonGroup(Ni,[rate_in]*Ni,clk)
neuron=NeuronGroup(1,(dv,dge,dgi),threshold=Vt,reset=Vr,clock=clk)
neuron.S[0]=Vr
neuron.S[1]=0.
neuron.S[2]=0.
Cex=STDPConnection(input_ex,neuron,1)
#Cex.connect(input_ex,neuron,gmax*rand(Ne,1))
Cex.connect_full(input_ex,neuron,weight=gmax*.5)
Cex.set_params(taup,taum,Ap,Am,gmax)
Cin=Connection(input_in,neuron,2)
Cin.connect_full(input_in,neuron,weight=gin)
M=SpikeMonitor(neuron)

net=Network([input_ex,input_in,neuron],[Cex,Cin,M])

print "Starting simulation..."
start_time=time.time()
for n in range(30):
    net.run(1*second)
    print M.nspikes
    l=[Cex.W.data[i][0] for i in range(Ne)]
    print hist(l)
    M.reinit()
print "Simulation done."
print "Simulation time:",time.time()-start_time,"seconds"
