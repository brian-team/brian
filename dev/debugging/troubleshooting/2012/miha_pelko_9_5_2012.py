from brian import *
from time import time

N = 100
M = 100
duration = 10*second

st = [[t*second for t in sorted(rand(M)*duration)] for _ in xrange(N)]

G = MultipleSpikeGeneratorGroup(st)
#G = SpikeGeneratorGroup(N, [(i, t) for i in xrange(len(st)) for t in st[i]])

start = time()
run(1*ms)
construct = time()-start
start = time()
run(duration-1*ms, report='stderr')
print 'Run time', time()-start
print 'Construction time', construct
