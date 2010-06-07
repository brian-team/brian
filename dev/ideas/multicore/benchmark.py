'''
Testing multiprocessing fro Brian
'''
from brian import *
from multiprocessing import Pool
from time import time

A=rand(3, 3)
X1=rand(3, 1000)
X2=rand(3, 1000)

def f(X):
    return dot(A, X)

p=Pool(2)
t1=time()
p.map(f, [X1, X2])
t2=time()
print t2-t1
