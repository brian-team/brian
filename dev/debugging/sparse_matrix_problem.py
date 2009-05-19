from brian import *
from scipy.sparse import lil_matrix
G = NeuronGroup(10, 'V:1')
H = NeuronGroup(10, 'V:1')
C = Connection(G, H, 'V')
C[:,0] = ones(10)
x = lil_matrix((10,10))
x[:,0] = ones(10)
print x.rows
print x.data
print C.W.rows
print C.W.data
run(1*ms)