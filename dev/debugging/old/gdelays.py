# Delays arising from interaction of synaptic conductances
# The idea: changing ge changes the delay
# Second idea: synaptic depression/facilitation changes the delay too
from brian import *

taum=20*ms
Ee=70*mV # relative to rest;
Ei=0*mV # shunting inhibition
taue=5*ms
taui=10*ms
N0=10
N=N0*N0

eqs='''
dv/dt=(-v+ge*(Ee-v)+gi*(Ei-v))/taum :  volt
dge/dt=-ge/taue : 1 # relative conductance
dgi/dt=-gi/taui : 1 # relative conductance
'''

neurons=NeuronGroup(N, model=eqs)
ge_range=linspace(0, 5, N0)
gi_range=linspace(0, 5, N0)
neurons.ge=array(ge_range*ones((N0, N0))).reshape(N)
neurons.gi=array(gi_range*ones((N0, N0))).T.reshape(N)

mon=StateMonitor(neurons, 'v', record=True) # that's heavy!

run(100*ms)

m=zeros(N)
for i in range(N):
    m[i]=mon.times[argmax(mon[i])]/ms

figure()
hist(m, 10)
figure()
pcolor(array(m).reshape(N0, N0))
#contour(array(m).reshape(N0,N0))
xlabel('ge')
ylabel('gi')
figure()
plot(mon.times/ms, mon[2*N0+2]/mV)
show()
