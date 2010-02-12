from brian import *

set_global_preferences(useweave=False)

G = NeuronGroup(3, 'V:1')
H = NeuronGroup(4, 'V:1')
C = Connection(G, H, 'V')
C.connect_random(p=0.5, weight=1)

C.compress()

print C.W.todense()
for i in range(C.W.shape[0]):
    print C.W[i, :].todense()
print
print C.W.todense().T
for i in range(C.W.shape[1]):
    print C.W[:, i].todense()