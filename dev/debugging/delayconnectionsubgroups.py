from brian import *

G = NeuronGroup(4, 'V:1')
Ga = G.subgroup(2)
Gb = G.subgroup(2)

C = Connection(G, G, 'V', delay=True, structure='dense')

C.connect_full(Ga, Gb, weight=1, delay=lambda:rand())

print C.W.todense()
print C.delayvec.todense()
