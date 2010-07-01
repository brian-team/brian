from brian import *
from scipy import sparse

P=NeuronGroup(1000,model='v:volt')
C=Connection(P,P,sparseness=0.1,weight=1*mV)
C2=Connection(P,P[0])
for i in range(len(P)):
    #print C[i,:].sum() # but sum(C[i,:]) does not work
    C2[i,0]=C[i,:].sum()

W=sparse.lil_matrix((5,5))
print W[1,:].sum() # same problem for sum(W[1,:])
