import multiprocessing
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

class ClusterManager(object):
    def __init__(self, work_class, shared_data, machines=[],
                 port=2718, authkey='brian cluster tools'):
        self.work_class = work_class
        self.port = port
        self.authkey = authkey
        # The first machine is the manager computer which can do work
        self.thismachine = ClusterMachine(work_class,
                                          shared_data=shared_data,
                                          port=port, authkey=authkey)
        # Generate clients
        self.clients = [Client((address, port),
                               authkey=authkey) for address in machines]
        # Send them each a copy of the shared data
        for client in self.clients:
            client.send(shared_data)
        # Get info about how many processors they have
        self.clients_info = [client.recv() for client in self.clients]
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
        min_gpu = min(self.num_gpu)
        if min_gpu>0:
            use_gpu = True
            self.num_processes = self.num_gpu
        else:
            use_gpu = False
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
                 port=2718, authkey='brian cluster tools'):
        self.work_class = work_class
        self.port = port
        self.authkey = authkey
        if have_gpu:
            self.num_gpu = drv.Device.count()
        else:
            self.num_gpu = 0
        self.num_cpu = multiprocessing.cpu_count()
        if shared_data is None:
            self.remote_machine = True
            address = ('localhost', port)
            self.listener = Listener(address, authkey=authkey)
            self.conn = self.listener.accept()
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
        self.server_conns, self.client_conns = zip(*self.pipes)
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
    return shared_data # for the moment we do nothing

def make_numpy(common_shared_data):
    return common_shared_data

def cluster_worker(common_shared_data, conn, process_number, use_gpu,
                   work_class):
    shared_data = make_numpy(common_shared_data)
    work_object = work_class(shared_data)
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
    