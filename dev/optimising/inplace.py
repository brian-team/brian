from numpy import *
from numpy.random import *
import time

N = 150

bx = randn(N, N)
by = randn(N, N)

t = time.time()
for i in xrange(1000):
    x = array(bx, copy=True)
    y = array(by, copy=True)
baset = time.time() - t

t = time.time()
for i in xrange(1000):
    x = matrix(bx, copy=True)
    y = matrix(by, copy=True)
basebt = time.time() - t

t = time.time()
for i in xrange(1000):
    x = array(bx, copy=True)
    y = array(by, copy=True)
    z = dot(x, y)
s = time.time() - t
print s, s - baset

t = time.time()
for i in xrange(1000):
    x = matrix(bx, copy=True)
    y = matrix(by, copy=True)
    z = x * y
s = time.time() - t
print s, s - basebt

t = time.time()
for i in xrange(1000):
    x = matrix(bx, copy=True)
    y = matrix(by, copy=True)
    x *= y
s = time.time() - t
print s, s - basebt
