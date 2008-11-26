from brian import *
import brianlib as bl

G = NeuronGroup(5, 'v:1')
H = NeuronGroup(3, 'v:1')
C = Connection(G, H, structure='dense')
C.W[:] = randn(*C.W.shape)

blG = bl.NeuronGroup(G._S, None, None, None, G.LS.S.n, G.LS.ind.n)
blH = bl.NeuronGroup(H._S, None, None, None, H.LS.S.n, H.LS.ind.n)
blC_W = bl.DenseConnectionMatrix(C.W)
blC = bl.Connection(blG, blH, blC_W)
x = arange(5)
#blC_W.add_row(0, H._S[0])
#blC_W.add_rows(x, H._S[0])
blC.propagate(x)

print C.W
print H._S
print sum(C.W, axis=0)