from brian import *
from scipy.sparse import coo_matrix
import gc
import random
import numpy
from scipy import weave
from time import time

if __name__=='__main__':
    usenew = False
    n = 100000
    m = 4000000
    seed(12321312)
    random.seed(340832)
    G = PoissonGroup(n, rates=50*Hz)
    H = NeuronGroup(n, 'v:1')
    
    i = randint(n, size=m)
    j = randint(n, size=m)
    w = rand(m)
    x = coo_matrix((w, (i, j)), shape=(n, n))
    #y = x.tocsr()
    #y = x.tocsr()
    start = time()
    if usenew:
        C = Connection(G, H, 'v')
        C.connect_from_sparse(x)
        print 'New time:',
    else:
        C = Connection(G, H, 'v', weight=x.tolil())
        C.compress()
        print 'Old time:',
    print time()-start
    raw_input()
    