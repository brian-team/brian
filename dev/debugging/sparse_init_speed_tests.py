from brian import *
import time

# Timing of algorithms
#N = 4000
#p = 0.1
#print '(N, p):', (N, p)
#
#G = NeuronGroup(N,'V:1')
#C = Connection(G, G, sparseness=p)
#start = time.time()
#C.compress()
#print 'Time:', time.time()-start

# verification of algorithms
N = 20
M = 50
p = 0.2
worked = True
for _ in range(100):    
    G = NeuronGroup(N, 'V:1')
    H = NeuronGroup(M, 'V:1')
    C = Connection(G, H, sparseness=p, weight=lambda i,j:randn())
    W = C.W.todense()
    C.compress()
    for i in range(M):
        if amax(abs(C.W[:,i].todense() - W[:,i].flatten()))>0.0001:
            worked = False
print 'Works:', worked