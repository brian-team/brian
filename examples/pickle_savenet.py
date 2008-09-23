'''
Pickling example, see also pickle_loadnet.py
'''
from brian import *
import pickle

clk = Clock()
eqs='''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''
P=NeuronGroup(4000,model=eqs,threshold=-50*mV,reset=-60*mV)
P.v=-60*mV
Pe=P.subgroup(3200)
Pi=P.subgroup(800)
Ce=Connection(Pe,P,'ge')
Ci=Connection(Pi,P,'gi')
Ce.connect_random(Pe, P, 0.02,weight=1.62*mV)
Ci.connect_random(Pi, P, 0.02,weight=-9*mV)
M=SpikeMonitor(P)
net = Network(P, Ce, Ci, M)
net.run(100*ms)

obj = (clk, eqs, P, Pe, Pi, Ce, Ci, M, net)

output = open('data.pkl', 'wb')
pickle.dump(obj,output,-1)
output.close()

# show what we've computed so far...
raster_plot(M)
show()