from numpy import *
import multiprocessing
import time
from multiprocessing import sharedctypes
from numpy import ctypeslib
import ctypes

def f(x, a, n):
    s=0
    for _ in xrange(n):
        s+=sum(x)*a
    return s

def process_listener(x, conn):
    x=ctypeslib.as_array(x)
    while True:
        conn.poll()
        a, n=conn.recv()
        conn.send(f(x, a, n))

if __name__=='__main__':
    N=10000000
    numprocesses=6
    complexity=10

    print 'Array size:', N
    print 'Num processors:', numprocesses
    print 'Task complexity:', complexity
    print

    x=ones(N)
    x_sct=sharedctypes.Array(ctypes.c_double, x, lock=False)
    A=1+arange(numprocesses)

    pipes=[multiprocessing.Pipe() for _ in xrange(numprocesses)]
    server_conns, client_conns=zip(*pipes)
    processes=[multiprocessing.Process(
                            target=process_listener,
                            args=(x_sct, conn)
                            ) for conn in client_conns]
    for p in processes:
        p.start()

    start=time.time()
    for i, a in enumerate(A):
        server_conns[i].send((a, complexity))
    for i in xrange(numprocesses):
        server_conns[i].poll()
        server_conns[i].recv()
    end=time.time()
    t_multiprocessing_1=end-start

    start=time.time()
    for i, a in enumerate(A):
        server_conns[i].send((a, complexity))
    for i in xrange(numprocesses):
        server_conns[i].poll()
        server_conns[i].recv()
    end=time.time()
    t_multiprocessing_2=end-start

    for p in processes:
        p.terminate()

    start=time.time()
    for a in A:
        f(x, a, complexity)
    end=time.time()
    t_single=end-start

    print 'Single:', t_single
    print 'Multiprocessing 1:', t_multiprocessing_1
    print 'Multiprocessing 2:', t_multiprocessing_2
    print 'S/M1:', t_single/t_multiprocessing_1
    print 'S/M2:', t_single/t_multiprocessing_2
