from numpy import split, concatenate, ndarray, cumsum, ndim, isscalar
from clustertools import *

all = ['DistributedFunction', 'distributed_worker']

def zip2(list, hole = None):
    """
    Zips several lists into a list of tuples, extending each tuple
    to the maximum size of the sub-lists.
    
    ``l`` is a list of lists of different sizes.
    ``zip2(l)`` returns a list of tuples, with ``hole`` values
    when there are holes.
    Like ``zip`` but doesn't truncate to the shortest sublist.
    """
    z = []
    k = 0
    b = True
    while b:
        b = False
        tz = []
        for l in list:
            if k >= len(l):
                tz.append(hole)
            else:
                b = True
                tz.append(l[k])
        z.append(tz)
        k += 1
    del z[-1]
    return z

class DistributedWorker:
    """
    Worker class for the ClusterManager object.
    
    Simply calls the function stored in ``shared_data['_fun']``
    with the arguments stored in ``job``.
    """
    def __init__(self, shared_data, use_gpu):
        self.shared_data = shared_data
    
    def process(self, job):
        # if shared_data only contains _fun, it means that the function
        # doesn't use shared_data.
        if len(self.shared_data)>1:
            result = self.shared_data['_fun'](job, self.shared_data)
        else:
            result = self.shared_data['_fun'](job)
        return result

class DistributedFunction():
    def __init__(self,  fun = None,
                        shared_data = None,
                        endaftercall = True,
                        machines = [],
                        gpu_policy = 'no_gpu',
                        max_cpu = None,
                        max_gpu = None,
                        named_pipe = None,
                        port = None,
                        accept_lists = False,# Set to True if the provided function handles a list as a parameter
                        verbose = False,
                        ):
        """
        Defines a distributed function from any function, allowing to execute it 
        transparently in parallel over several workers (multiple CPUs on a single machine
        or several machines connected in a network). 

        **Arguments**
        
        ``fun``
            The Python function to parallelize. There are two different ways of 
            parallelizing a function.
            * If ``fun`` accepts a single argument and returns a single object,
              then the distributed function can be called with a list of arguments
              that are transparently spread among the workers.
            * If ``fun`` accepts any D*N matrix and returns a N-long vector,
              in such a way that the computation is performed column-wise
              (that is, there exists a function f : R^d -> R such that
              ``fun(x)[i] == f(x[:,i])``), then the distributed function can be 
              called exactly like the original function ``fun`` : each worker 
              will call ``fun`` with a view on ``x``. The results are then 
              automatically concatenated, so that using the distributed
              function is strictly equivalent to using the original function.
        """
        if fun is None:
            raise Exception('The function must be provided')
        
        if shared_data is None:
            shared_data = dict([])
        
        self.endaftercall = endaftercall
        self.verbose = verbose
        shared_data['_fun'] = fun
        self.manager = ClusterManager(DistributedWorker, 
                                      shared_data = shared_data,
                                      machines = machines,
                                      gpu_policy = gpu_policy,
                                      own_max_cpu = max_cpu,
                                      own_max_gpu = max_gpu,
                                      named_pipe = named_pipe,
                                      port = port,
                                      authkey = 'distopt') 
        self.numprocesses = self.manager.total_processes
        self.accept_lists = accept_lists
        
        if verbose:
            # Displays the number of cores used
            if self.manager.use_gpu:
                cores =  'GPU'
            else:
                cores = 'CPU'
            if self.numprocesses > 1:
                b = 's'
            else:
                b = ''
            print "Using %d %s%s..." % (self.numprocesses, cores, b)

    def divide(self, n):
        worker_size = [n/self.numprocesses for _ in xrange(self.numprocesses)]
        worker_size[-1] = int(n-sum(worker_size[:-1]))
        
        bins = [0]
        bins.extend(list(cumsum(worker_size)))
        
        return worker_size, bins

    def prepare_jobs(self, x):
        ncalls = 1
        if x is None:
            jobs = [None for _ in xrange(self.numprocesses)]
        elif isinstance(x, list):
            n = len(x)
            # Nothing to do if x is smaller than numprocesses
            if n <= self.numprocesses:
                jobs = x
            else:
                # If the function handles lists, then divide the list into sublists
                if self.accept_lists:
                    worker_size, bins = self.divide(n)
                    jobs = [x[bins[i]:bins[i+1]] for i in xrange(self.numprocesses)]
                # Otherwise, performs several calls of the function on each worker
                # for each element in the list (default case)
                else:
                    jobs = []
                    job = []
                    for i in xrange(len(x)):
                        if len(job) >= self.numprocesses:
                            jobs.append(job)
                            job = []
                            ncalls += 1
                        job.append(x[i])
                    if len(job)>0:
                        jobs.append(job)
                    else:
                        ncalls -= 1
        elif isinstance(x, ndarray):
            d = ndim(x)
            if d == 0:
                jobs = [None for _ in xrange(self.numprocesses)]
            else:
                n = x.shape[-1]
                worker_size, bins = self.divide(n)
                bins = bins[1:-1]
                jobs = split(x, bins, axis=-1)
        elif isinstance(x, dict):
            jobs = [dict([]) for _ in xrange(self.numprocesses)]
            for param, value in x.iteritems():
                subjobs = self.prepare_jobs(value)
                for i in xrange(self.numprocesses):
                    jobs[i][param] = subjobs
        return jobs, ncalls

    def __call__(self, x = None):
        jobs, ncalls = self.prepare_jobs(x)
        if ncalls == 1:
            results = self.manager.process_jobs(jobs)
        else:
            if self.verbose:
                print "Using %d successive function calls on each worker..." % ncalls
            results = []
            for subjobs in jobs:
                results.extend(self.manager.process_jobs(subjobs))
        if isinstance(x, ndarray):
            results = concatenate(results, axis=-1)
        if self.endaftercall:
            self.end()
        return results
    
    def __del__(self):
        self.end()
        
    def end(self):
        self.manager.finished()

def distributed_worker(max_cpu = None, max_gpu = None, port = None,
                      named_pipe = None):
    cluster_worker_script(DistributedWorker,
                          max_gpu=max_gpu, max_cpu=max_cpu, port=port,
                          named_pipe=named_pipe, authkey='distopt')

