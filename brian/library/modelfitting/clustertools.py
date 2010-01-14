from multiprocessing import sharedctypes
from brian import Equations
import numpy
from numpy import ctypeslib
import ctypes
import gc
import time
import multiprocessing
import cPickle
import zlib
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

class ClusterConnection(object):
    '''
    Handles chunking and compression of data.
    
    To minimise data transfers between machines, we can use data compression,
    which this Connection handles automatically.
    
    Windows named pipes are limited to 64k writes at any one time, due to
    a bug (?) in Python's multiprocessing, this means we can only send a
    maximum of 64k of data in any one send() or recv() operation. This
    version of multiprocessing.Connection can split data into 64k chunks, which
    should be used only if using Windows named pipes.
    '''
    def __init__(self, conn, chunked=False, compressed=True):
        self.conn = conn
        self.chunked = chunked
        self.compressed = compressed
        self.BUFSIZE = 65500
    def send(self, obj):
        s = cPickle.dumps(obj, -1)
        if self.compressed:
            s = zlib.compress(s)
        if self.chunked:
            l = 1+len(s)//self.BUFSIZE
            self.conn.send(l)
            for i in xrange(l):
                self.conn.send(s[i*self.BUFSIZE:(i+1)*self.BUFSIZE])
        else:
            self.conn.send(s)
    def recv(self):
        start = time.time()
        if self.chunked:
            l = self.conn.recv()
            data = []
            for i in xrange(l):
                data.append(self.conn.recv())
            s = ''.join(data)
        else:
            s = self.conn.recv()
        if self.compressed:
            s = zlib.decompress(s)
        end = time.time()
        #print 'Recv:', end-start
        return cPickle.loads(s)
    def poll(self, *args, **kwds):
        return self.conn.poll(*args, **kwds)
    def close(self):
        return self.conn.close()

class ClusterManager(object):
    '''
    ClusterManager is used for managing a cluster.
    
    To use the clustertools module, you have to do the following:
    
    * Write a class work_class as described below, this is the user
      class in charge of doing the actual task.
    * Create shared_data that each process can read from but not write
      to.
    * Run worker machines using cluster_worker_script.
    * Create a ClusterManager
    * Submit jobs to ClusterManager
    * Close the ClusterManager connection
    
    In addition, you can specify the number of GPUs and CPUs you
    want to use on this computer with own_max_gpu and own_max_cpu.
    You can also specify a policy of how to deal with mixed GPUs and
    CPUs (see policies below).
    
    The machines argument should be a list of host/IP names (if using IP)
    or machine names (if using Windows named pipes). If IP is being used
    you can specify a port with the port keyword. Set named_pipe=True if
    using named pipes, or to a string to use a particular named pipe.
    Set authkey to a shared password for authentication. For named pipes,
    note that the user of the manager computer has to have a logon on
    each worker with the same ID and password.
    
    **work_class**
    
    Work class should follow the following template::

        class work_class(object):
            def __init__(self, shared_data, use_gpu):
                ...
            def process(self, job):
                ...
    
    The __init__ method is called on creation by cluster tools with
    the shared data specified, and whether or not to use the GPU if
    available (following the GPU policy). The process method is called
    with each job submitted and should return some values. Note that
    all shared_data, job and return values from process should be
    picklable.
    
    **shared_data**
    
    Shared data is read-only. It should be a dictionary, whose values
    are picklable. If the values are numpy arrays, and the data is being
    shared to processes on a given computer, the memory will not be
    copied, but a pointer passed to the child processes, saving memory.
    Large read-only data to be shared should be put in here.
    
    **GPU policies**
    
    The policies are 'prefer_gpu' (default) which will use only GPUs if
    any are available on any of the computers, 'require_all' which will
    only use GPUs if all computers have them, or 'no_gpu' which will
    never use GPUs even if available.
    
    **Using the ClusterManager object**
    
    After the object is initialised, it has an attribute:
    
    ``total_processes``
        The total number of processes on the cluster, so that you can 
        divide work up appropriately.
    
    and methods:
    
    ``process_jobs(jobs)``
        This will process a list of jobs, where the size of the list
        should be ``total_processes``.
    ``finished()``
        Should be called after all jobs have been finished.
    '''
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
        import time
        self.clients = [Client(address,
                               authkey=authkey) for address in machines]
        if named_pipe is not None:
            chunked = True
        else:
            chunked = False
        self.clients = [ClusterConnection(client, chunked=chunked) for client in self.clients]
        # Send them each a copy of the shared data
        start = time.time()
        for client in self.clients:
            client.send(shared_data)
        # Get info about how many processors they have
        self.clients_info = [client.recv() for client in self.clients]
        end = time.time()
        #print 'Data transfer took:', end-start
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
    '''
    ClusterMachine should be called on each worker machine to receive
    jobs. After it receives the finished notification, it will shut
    down. To repeatedly run ClusterMachine instances, use the
    cluster_worker_script function.
    
    ClusterMachine has a keyword for shared_data but you should not
    provide a value for it, this is done automatically. You should
    provide a work_class object.
    
    You can also specify max_gpu, max_cpu, port, named_pipe and authkey
    as in ClusterManager.
    '''
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
            if named_pipe is not None:
                chunked = True
            else:
                chunked = False
            self.conn = ClusterConnection(self.conn, chunked=chunked)
            self.shared_data = self.conn.recv()
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
        set_gpu_device(drv.Device.count()-process_number-1)
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
    '''
    Call this on worker machines, the arguments and keywords are
    passed to ClusterMachine (see there for details).
    '''
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
            
        