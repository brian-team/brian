import time
from numpy import *
from numpy.random import *
from fastexp import *

x = array([1, 2, 3], dtype=float)
y = exp(x)
z = zeros(len(x))
fastexp(x, z)
print y
print z

x = randn(10000000)
y = empty(x.shape)

t = time.time()
z = exp(x)
print "exp", time.time() - t

t = time.time()
fastexp(x, y)
print "fastexp", time.time() - t
