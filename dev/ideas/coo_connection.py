from brian import *
from scipy.sparse import coo_matrix
import gc
import random
import numpy
from scipy import weave


if __name__=='__main__':
    seed(12321312)
    random.seed(340832)
    n = 10
    m = 40
    i = randint(n, size=m)
    j = randint(n, size=m)
    w = rand(m)
    x = coo_matrix((w, (i, j)), shape=(n, n))
    print x.todense()
    y = x.tocsr()
    y = x.tocsr()
    G = PoissonGroup(n, rates=50*Hz)
    H = NeuronGroup(n, 'v:1')
    C = Connection(G, H, 'v')
    #C = Connection(G, H, 'v', weight=x.tolil())
    M = StateMonitor(H, 'v', record=[0,1,2])
    #set_connection_from_sparse(C, x, column_access=True)
    C.connect_from_sparse(x)
    print C.W.allj.dtype
    run(100*ms)
    print sum(M.values) # 9186.73073667
#    M.plot()
#    show()
    