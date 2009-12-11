from numpy import *
import multiprocessing
import time
from multiprocessing import sharedctypes
from numpy import ctypeslib
import ctypes

def f(x, a, n):
    s = 0
    for _ in xrange(n):
        s += sum(x)*a
    return s

def process_listener(x, task_queue, done_queue):
    x = ctypeslib.as_array(x)
    for task in iter(task_queue.get, 'STOP'):
        a, n = task
        done_queue.put(f(x, a, n))

if __name__=='__main__':
    N = 10000000
    numprocesses = 6
    complexity = 10

    print 'Array size:', N
    print 'Num processors:', numprocesses
    print 'Task complexity:', complexity
    print

    x = ones(N)
    x_sct = sharedctypes.Array(ctypes.c_double, x, lock=False)
    A = 1+arange(numprocesses)

    task_queues = [multiprocessing.Queue() for _ in xrange(numprocesses)]
    done_queues = [multiprocessing.Queue() for _ in xrange(numprocesses)]
    processes = [multiprocessing.Process(
                            target=process_listener,
                            args=(x_sct, tasks, done)
                            ) for tasks, done in zip(task_queues, done_queues)]
    for p in processes:
        p.start()
    
    start = time.time()
    for i, a in enumerate(A):
        task_queues[i].put((a, complexity))
    for i in xrange(numprocesses):
        done_queues[i].get()
    end = time.time()
    t_multiprocessing_1 = end-start

    start = time.time()
    for i, a in enumerate(A):
        task_queues[i].put((a, complexity))
    for i in xrange(numprocesses):
        done_queues[i].get()
    end = time.time()
    t_multiprocessing_2 = end-start
    
    for p in processes:
        p.terminate()
    
    start = time.time()
    for a in A:
        f(x, a, complexity)
    end = time.time()
    t_single = end-start

    print 'Single:', t_single
    print 'Multiprocessing 1:', t_multiprocessing_1
    print 'Multiprocessing 2:', t_multiprocessing_2
    print 'S/M1:', t_single/t_multiprocessing_1    
    print 'S/M2:', t_single/t_multiprocessing_2
    