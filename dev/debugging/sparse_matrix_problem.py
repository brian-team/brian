from brian import *
from scipy.sparse import lil_matrix
G = NeuronGroup(10, 'V:1')
H = NeuronGroup(5, 'V:1')
C = Connection(G, H, 'V')
C[:,1] = arange(10)
print C.W.rows
print C.W.data
print C.W.todense()
run(1*ms)