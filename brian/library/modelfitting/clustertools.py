from multiprocessing import sharedctypes
from brian import Equations
import numpy
from numpy import ctypeslib
import ctypes
import gc
import multiprocessing
import cPickle
from multiprocessing.connection import Listener, Client
try:
    import pycuda
    import pycuda.autoinit as autoinit
    import pycuda.driver as drv
    def set_gpu_device(n):
        '''
        This function makes pycuda use GPU number n in the system.
        '''
        global _gpu_context # we make this a global variable so that a
                            # reference is kept to it and the context is not
                            # deallocated
        autoinit.context.pop() # delete the old context before making a new one
        _gpu_context = drv.Device(n).make_context()
    have_gpu = True
except ImportError:
    have_gpu = False

__all__ = ['ClusterManager', 'ClusterMachine', 'cluster_worker_script']

gpu_policies = {
    'prefer_gpu':lambda ncpus, ngpus: numpy.amax(ngpus)>0,
    'require_all':lambda ncpus, ngpus: numpy.amin(ngpus)>0,
    'no_gpu':lambda ncpus, ngpus: False,
    }

class ChunkedConnection(object):
    def __init__(self, conn):
        self.conn = conn
        self.BUFSIZE = 65500
    def send(self, obj):
        s = cPickle.dumps(obj, -1)
        l = len(s)//self.BUFSIZE
        self.conn.send(l)
        for i in xrange(l):
            self.conn.send(s[i*self.BUFSIZE:(i+1)*self.BUFSIZE])
    def recv(self):
        l = self.conn.recv()
        data = []
        for i in xrange(l):
            data.append(self.conn.recv())
        s = ''.join(data)
        return cPickle.loads(s)

class ClusterManager(object):
    def __init__(self, work_class, shared_data, machines=[],
                 own_max_gpu=None, own_max_cpu=None,
                 gpu_policy='prefer_gpu',
                 port=None, named_pipe=None,
                 authkey='brian cluster tools'):
        
        self.work_class = work_class
        if port is None and named_pipe is None:
            port = 2718
        if named_pipe is True:
            named_pipe = 'BrianModelFitting'
        self.port = port
        self.named_pipe = named_pipe
        self.authkey = authkey
        if isinstance(gpu_policy, str):
            gpu_policy = gpu_policies[gpu_policy]
        # The first machine is the manager computer which can do work
        self.thismachine = ClusterMachine(work_class,
                                          shared_data=shared_data,
                                          max_gpu=own_max_gpu,
                                          max_cpu=own_max_cpu,
                                          port=port, named_pipe=named_pipe,
                                          authkey=authkey)
        # Generate clients
        if port is not None and named_pipe is None:
            machines = [(address, port) for address in machines]
        elif named_pipe is not None and port is None:
            machines = ['\\\\'+address+'\\pipe\\'+named_pipe for address in machines]
        self.clients = [Client(address,
                               authkey=authkey) for address in machines]
        self.clients = [ChunkedConnection(client) for client in clients]
        # Send them each a copy of the shared data
        for client in self.clients:
            print 'Sending data'
            import pickle
#                    s = self._dumps(obj)
#        self._conn.send_bytes(s)
            s = pickle.dumps(shared_data,-1)
            print 'Data length:', len(s)
            client.send(shared_data)
            print 'Sent data'
        # Get info about how many processors they have
        print 'Receiving data from clients'
        self.clients_info = []
        for client in self.clients:
            while True:
                if client.poll(10):
                    print 'Polled data'
                    self.clients_info.append(client.recv())
                    print 'Received data'
                    break
                else:
                    print 'Still waiting on client'
        #self.clients_info = [client.recv() for client in self.clients]
        print 'Received clients_info'
        if len(self.clients_info):
            self.num_cpu, self.num_gpu = zip(*self.clients_info)
            self.num_cpu = list(self.num_cpu)
            self.num_gpu = list(self.num_gpu)
        else:
            self.num_cpu = []
            self.num_gpu = []
        self.num_cpu.append(self.thismachine.num_cpu)
        self.num_gpu.append(self.thismachine.num_gpu)
        # Decide whether to use GPUs or CPUs (only use CPUs if not all
        # computers have GPUs)
        if gpu_policy(self.num_cpu, self.num_gpu):
            self.use_gpu = use_gpu = True
            self.num_processes = self.num_gpu
        else:
            self.use_gpu = use_gpu = False
            self.num_processes = self.num_cpu
        for client in self.clients:
            client.send(use_gpu)
        self.thismachine.prepare_workers(use_gpu)
        self.total_processes = sum(self.num_processes)
    def process_jobs(self, jobs):
        n = len(jobs)-self.thismachine.num_processes
        for client, num_processes in zip(self.clients, self.num_processes):
            client.send(jobs[n-num_processes:n])
        myresults = self.thismachine.process_jobs(jobs[-self.thismachine.num_processes:])
        results = []
        for client in self.clients:
            results.extend(client.recv())
        results.extend(myresults)
        return results
    def finished(self):
        self.thismachine.finished()
        for client in self.clients:
            client.send(None)
            client.close()

class ClusterMachine(object):
    def __init__(self, work_class, shared_data=None,
                 max_gpu=None, max_cpu=None,
                 port=None, named_pipe=None,
                 authkey='brian cluster tools'):
        self.work_class = work_class
        if port is None and named_pipe is None:
            port = 2718
        if named_pipe is True:
            named_pipe = 'BrianModelFitting'
        self.port = port
        self.named_pipe = named_pipe
        self.authkey = authkey
        if have_gpu:
            gpu_count = drv.Device.count()
            if max_gpu is None:
                max_gpu = gpu_count
            self.num_gpu = min(max_gpu, gpu_count)
        else:
            self.num_gpu = 0
        cpu_count = multiprocessing.cpu_count()
        if max_cpu is None:
            max_cpu = cpu_count
        self.num_cpu = min(max_cpu, cpu_count)
        if shared_data is None:
            self.remote_machine = True
            if port is not None and named_pipe is None:
                address = ('localhost', port)
            elif port is None and named_pipe is not None:
                address = '\\\\.\\pipe\\'+named_pipe
            self.listener = Listener(address, authkey=authkey)
            self.conn = self.listener.accept()
            self.conn = ChunkedConnection(self.conn)
            while True:
                if self.conn.poll(10):
                    print 'Polled data'
                    self.shared_data = self.conn.recv()
                    print 'Received data'
                    break
                else:
                    print 'Still waiting.'
            # Send a message to the manager telling it the number of available
            # CPUs and GPUs
            self.conn.send((self.num_cpu, self.num_gpu))
        else:
            self.remote_machine = False
            self.jobs = []
            self.shared_data = shared_data
        self.common_shared_data = make_common(self.shared_data)
        if self.remote_machine:
            # find out whether we are using GPUs or not and prepare workers
            self.prepare_workers(self.conn.recv())
            # and enter job queue
            while True:
                jobs = self.conn.recv()
                if jobs is None:
                    break
                self.conn.send(self.process_jobs(jobs))
    def prepare_workers(self, use_gpu):
        self.use_gpu = use_gpu
        if use_gpu:
            self.num_processes = self.num_gpu
        else:
            self.num_processes = self.num_cpu
        self.pipes = [multiprocessing.Pipe() for _ in xrange(self.num_processes)]
        if len(self.pipes):
            self.server_conns, self.client_conns = zip(*self.pipes)
        else:
            self.server_conns = []
            self.client_conns = []
        self.processes = [multiprocessing.Process(
                                target=cluster_worker,
                                args=(self.common_shared_data, conn, n,
                                      self.use_gpu, self.work_class)
                                ) for n, conn in enumerate(self.client_conns)]
        for p in self.processes:
            p.start()
    def process_jobs(self, jobs):
        for conn, job in zip(self.server_conns, jobs):
            conn.send(job)
        return [conn.recv() for conn in self.server_conns]
    def finished(self):
        for conn in self.server_conns:
            conn.send(None)
        for p in self.processes:
            p.terminate()
        
# This function should turn arrays into sharedctypes ones to minimise
# data copying, assume shared_data is a dictionary of arrays and values
def make_common(shared_data):
    data = {}
    for k, v in shared_data.iteritems():
        if isinstance(v, numpy.ndarray):
            mapping = {
                numpy.float64:ctypes.c_double,
                numpy.int32:ctypes.c_int,
                }
            ctype = mapping.get(v.dtype, None)
            if ctype is not None:
                v = sharedctypes.Array(ctype, v, lock=False)
        data[k] = v
    return data

def make_numpy(common_shared_data):
    data = {}
    for k, v in common_shared_data.iteritems():
        if hasattr(v, 'as_array') and not isinstance(v, Equations):#isinstance(v, sharedctypes.Array):
            v = ctypeslib.as_array(v)
        data[k] = v
    return data

def cluster_worker(common_shared_data, conn, process_number, use_gpu,
                   work_class):
    shared_data = make_numpy(common_shared_data)
    work_object = work_class(shared_data, use_gpu)
    if use_gpu:
        set_gpu_device(process_number)
    while True:
        try:
            job = conn.recv()
        except EOFError:
            job = None
        if job is None:
            break
        conn.send(work_object.process(job))
    conn.close()

def cluster_worker_script(*args, **kwds):
    try:
        n = 0
        while True:
            print 'Job schedule', n, 'started.'
            machine = ClusterMachine(*args, **kwds)
            del machine
            gc.collect()
            n += 1
    except KeyboardInterrupt:
        try:
            for p in machine.processes:
                p.terminate()
            print 'Worker processes shut down successfully.'
        except:
            print 'Error shutting down worker processes, kill them manually.'
            
        