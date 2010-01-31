from brian import *
import multiprocessing
from numpy import ctypeslib
import ctypes
from multiprocessing import sharedctypes
import numpy

def worker(S, shape, i, j, A, C, conn):
    S = ctypeslib.as_array(S)
    S.shape = shape
    S = S[:, i:j]
    sys.stdout.flush()
    while True:
        try:
            job = conn.recv()
        except EOFError:
            job = None
        if job is None:
            break
        S[:] = dot(A, S)
        if C is not None:
            add(S, C, S)
        conn.send(True)
    conn.close()

class MulticoreLinearStateUpdater(LinearStateUpdater):
    def __init__(self, stateupdater, group, ncpu=None):
        self.stateupdater = stateupdater
        self.group = group
        if ncpu is None:
            ncpu = multiprocessing.cpu_count()
        self.ncpu = ncpu
        subgroup_sizes = [len(group)/ncpu]*ncpu
        subgroup_sizes[-1] += len(group)-sum(subgroup_sizes)
        I = hstack((0, cumsum(subgroup_sizes)))
        S = group._S
        shape = S.shape
        S = reshape(S, S.size)
        size = S.size
        S = sharedctypes.RawArray('d', S)
        # must replace group._S to use this shared memory!
        group._S = numpy.frombuffer(S, dtype=numpy.float64, count=size)
        group._S.shape = shape
        A = stateupdater.A
        if stateupdater._useB:
            C = stateupdater._C
        else:
            C = None
        self.pipes = [multiprocessing.Pipe() for _ in xrange(ncpu)]
        self.server_conns, self.client_conns = zip(*self.pipes)
        self.processes = [multiprocessing.Process(
                                target=worker,
                                args=(S, shape, i, j, A, C, conn)
                                ) for i, j, conn in zip(I[:-1],
                                                        I[1:],
                                                        self.client_conns)]
        for p in self.processes:
            p.start()
    def __call__(self, G):
        assert G is self.group
        for conn in self.server_conns:
            conn.send(True)
        return [conn.recv() for conn in self.server_conns]
    def __del__(self):
        for conn in self.server_conns:
            conn.send(None)
        for p in self.processes:
            p.terminate()
        
if __name__=='__main__':
    import time
    
    # timings on my dual-core Dell E4300 for N=40,000
    # with connections:
    # ncpu=1: 18.5
    # ncpu=2: 12.5
    # without connections:
    # ncpu=1: 17.5
    # ncpu=2: 12.9
    # without connections for N=400,000 and duration=.01*second:
    # ncpu=1: 28.1
    # ncpu=2: 31.2
    # without connections N=4,000 and duration=10*second:
    # ncpu=1: 1.6s
    # ncpu=2: 1.9s
    
    ncpu = None
    duration = 1*second
    N = 4000
    monitor = True
    have_connections = True
    
    Ni = int(N*.2)
    Ne = N-Ni
    p = 80.0/N
    
    eqs='''
    dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
    '''
    
    P=NeuronGroup(N,model=eqs,
                  threshold=-50*mV,reset=-60*mV)
    P.v=-60*mV+10*mV*rand(len(P))
    Pe=P.subgroup(Ne)
    Pi=P.subgroup(Ni)
    
    if have_connections:
        Ce=Connection(Pe,P,'ge')
        Ci=Connection(Pi,P,'gi')
        Ce.connect_random(Pe, P, p, weight=1.62*mV)
        Ci.connect_random(Pi, P, p, weight=-9*mV)
    
    if monitor:
        M=SpikeMonitor(P)
        Mv = StateMonitor(P, 'v', record=[0])

    if ncpu is None or ncpu>1:
        P._state_updater = MulticoreLinearStateUpdater(P._state_updater, P,
                                                       ncpu=ncpu)
    
    run(1*msecond)
    
    start = time.time()
    
    run(duration)
    
    end = time.time()
    
    print 'N:', N
    print 'duration:', duration
    print 'ncpu:', ncpu
    print 'Time taken:', end-start
    
    del P._state_updater

    if monitor:
        subplot(211)
        raster_plot(M)
        subplot(212)
        Mv.plot()
        show()
