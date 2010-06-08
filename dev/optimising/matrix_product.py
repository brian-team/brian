# Matrix product testing
from scipy import *
from scipy import weave
from time import time
from scipy import linalg
import numpy

gemm, = linalg.blas.get_blas_funcs(['gemm'])

iterations = 10000
N = 4000
m = 3

A = array(rand(m, m))
M = array(rand(m, N))
B = array(zeros((m, N)))
V = array(rand(m, 1))

t1 = time()
for _ in xrange(iterations):
    B = numpy.core._dotblas.dot(A, M)
    #B=gemm(1.0,A,M,0,M,0,0,overwrite_c=0)
t2 = time()
print t2 - t1
