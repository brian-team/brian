from brian import *

G = NeuronGroup(1, 'dv/dt=1/second:1')
H = NeuronGroup(1, 'dv/dt=1/second:1')

net = Network(G)
net.run(100*ms)
print G.v, H.v
net.add(H)
net.run(100*ms)
print G.v, H.v
net.remove(G)
net.run(500*ms)
print G.v, H.v
