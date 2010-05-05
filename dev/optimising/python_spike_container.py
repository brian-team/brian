#return self.X[range(i0,n)+range(0,j0)] # Costly operation!
##return concatenate((self.X[i0:],self.X[0:j0])) # might be slightly faster
##return self.X.take(range(i0,j0+n),mode='wrap') # does not seem to work

from pylab import *
import time, timeit
from numpy import zeros, hstack, concatenate

def f_hstack(n, m, repeats=100000):
    x = zeros(n)
    start = time.time()
    for _ in xrange(repeats):
        hstack((x[m:], x[:m]))
    end = time.time()
    return (end-start)/repeats

def f_concatenate(n, m, repeats=100000):
    x = zeros(n)
    start = time.time()
    for _ in xrange(repeats):
        concatenate((x[m:], x[:m]))
    end = time.time()
    return (end-start)/repeats

def f_range(n, m, repeats=100000):
    x = zeros(n)
    start = time.time()
    for _ in xrange(repeats):
        x[range(m, n)+range(m)]
    end = time.time()
    return (end-start)/repeats

def f_take(n, m, repeats=100000):
    x = zeros(n)
    start = time.time()
    for _ in xrange(repeats):
        x.take(arange(m, m+n), mode='wrap')
    end = time.time()
    return (end-start)/repeats

N = [1, 2, 3, 4, 5, 10, 100]
P = [.1, .25, .5, .75]

for p in P:
    print p
    t_hstack = []
    t_concatenate = []
    t_range = []
    t_take = []
    for n in N:
        print n
        m = int(n*p)
        t_hstack.append(f_hstack(n, m))
        t_concatenate.append(f_concatenate(n, m))
        t_range.append(f_range(n, m))
        t_take.append(f_take(n, m))
    plot(N, t_hstack, '-r', label=str(p))
    plot(N, t_concatenate, '-g', label=str(p))
    plot(N, t_range, '-b', label=str(p))
    plot(N, t_take, '-k', label=str(p))
#legend()
show()
