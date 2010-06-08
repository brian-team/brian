from numpy import *
import multiprocessing
import time
from multiprocessing import sharedctypes
from numpy import ctypeslib
import ctypes

def f(x, a, n):
    s = 0
    for _ in xrange(n):
        s += sum(a * x)
    return s

def test_with_copy_func((x, a, n)):
    return f(x, a, n)

def test_with_copy(N, M, complexity):
    x = ones(N)
    A = 1 + arange(M)
    pool = multiprocessing.Pool()
    args = [(x, a, complexity) for a in A]

    start = time.time()
    results = pool.map(test_with_copy_func, args)
    end = time.time()
    t_multiprocessing = end - start

    results = []
    start = time.time()
    for x, a, complexity in args:
        results.append(test_with_copy_func((x, a, complexity)))
    end = time.time()
    t_single = end - start

    print 'Test with copy'
    print '--------------'
    print 'Single:', t_single
    print 'Multiprocessing:', t_multiprocessing
    print 'S/M:', t_single / t_multiprocessing

def test_with_shared_func(x, a, complexity):
    x = ctypeslib.as_array(x)
    return f(x, a, complexity)

def test_with_shared(N, M, complexity):
    x = ones(N)
    x_sct = sharedctypes.Array(ctypes.c_double, x, lock=False)
    A = 1 + arange(M)

    #pool = multiprocessing.Pool()
    start = time.time()
    processes = [multiprocessing.Process(target=test_with_shared_func,
                                         args=(x_sct, a, complexity)) for a in A]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    end = time.time()
    t_multiprocessing = end - start

    results = []
    args = [(x, a, complexity) for a in A]
    start = time.time()
    for x, a, complexity in args:
        results.append(test_with_shared_func(x, a, complexity))
    end = time.time()
    t_single = end - start

    print 'Test with shared'
    print '----------------'
    print 'Single:', t_single
    print 'Multiprocessing:', t_multiprocessing
    print 'S/M:', t_single / t_multiprocessing

if __name__ == '__main__':
    N = 10000000
    M = 8
    complexity = 1
    print 'Array size:', N
    print 'Num processors:', M
    print 'Task complexity:', complexity
    print
    test_with_shared(N, M, complexity)
    print
    test_with_copy(N, M, complexity)
