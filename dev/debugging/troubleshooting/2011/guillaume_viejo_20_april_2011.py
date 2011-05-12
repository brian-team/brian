from brian import *

tau_c=100*ms
tau=10*ms
w0=1*mV

eqs_neurons='''
dv/dt=-v/tau : volt
'''

P=NeuronGroup(10,eqs_neurons,threshold=10*mV,reset=0*mV)
input=PoissonGroup(10,rates=10*Hz)
synapses=Connection(input,P,'v',weight=4*mV)

eqs_stdp='''
x : 1 # fictional presynaptic variable
dc/dt = -c/tau_c : 1 # your postsynaptic calcium variable
v : volt # a copy of the postsynaptic v
'''
stdp=STDP(synapses, eqs=eqs_stdp,pre="w+=w0*(c>1)*(v>5*mV);x",post="c+=1;v", wmax=10*mV)
stdp.post_group.v=linked_var(P,'v')

run(1000*ms)
