from numpy import split, concatenate, ndarray, cumsum, ndim, isscalar
from clustertools import *
import sys
#import logging

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
        
        Usage examples
        ==============
        
        Simple example 1
        ----------------
        
        The simplest way of using ``DistributedFunction`` is by defining a function which
        accepts a single object (a number, a Numpy array or any other object) as an argument 
        and returns a result. Using ``DistributedFunction`` allows to call this function in 
        parallel over multiple CPUs/machines with several objects as arguments, and retrieving 
        the result for each argument. By default, ``DistributedFunction`` uses all available CPUs
        in the system.
        
        In the following example which computes the inverse of two matrices, we assume that there 
        are at least two CPUs in the system. Each CPU computes the inverse of a single matrix::
            
            # For Windows users, it is required that any code using this library is placed after
            # this line, otherwise the system will crash!
            if __name__ == '__main__':
                
                from numpy import dot
                from numpy.random import rand
                from numpy.linalg import inv
                
                # Import the library to have access to the ``DistributedFunction`` class
                from distfun import *
                
                # We define the two matrices that are going to be inversed in parallel.
                A = rand(4,4)
                B = rand(4,4)
                
                # The first argument of ``DistributedFunction`` is the name of the function
                # that is being parallelized. It must accept a single argument and returns a
                # single object. The optional argument ``max_cpu`` allows to limit the number
                # of CPUs that are going to be used by the parallelized function. Of course,
                # it has no effect if there are less CPUs available in the system.
                distinv = DistributedFunction(inv, max_cpu=2)
                
                # ``distinv`` is the parallelized version of ``inv`` : it is called by passing
                # a list of arguments. The list can be of any size. If there are more arguments
                # than workers, then each worker will process several arguments in series.
                # Here, if there are two available CPUs in the system, the first CPU inverses
                # A, the second inverses B. ``invA`` and ``invB`` contain the inverses of A and B.
                invA, invB = distinv([A,B])
                
        Simple example 2
        ----------------
        
        If ``fun(x)`` is a Python function that accepts a Numpy array as argument ``x``,
        assuming that it
        
        from distfun import *
        dfun = DistributedFunction(fun)
        
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
        """
        Detects that at least one worker couldn't call the function.
        In this case, the library must call each worker several times.
        """
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

