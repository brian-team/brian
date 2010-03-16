from numpy import split, concatenate, ndarray, cumsum, ndim, isscalar
from clustertools import *
import sys
#import logging

all = ['DistributedFunction', 'distributedslave', 'InvalidArgument']

class InvalidArgument(Exception):
    def __init__(self):
        pass
    
def zip2(list, hole = None):
    """
    l is a list of lists of different sizes.
    zip2(l) returns a list of tuples, with 'hole' values
    when there are holes.
    Like zip but doesn't truncate to the shortest sublist
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
    def __init__(self, shared_data, use_gpu):
        self.shared_data = shared_data
    
    def process(self, job):
        if type(job) == InvalidArgument:
            return None
        try:
            if len(self.shared_data)>1:
                result = self.shared_data['_fun'](job, **self.shared_data)
            else:
                result = self.shared_data['_fun'](job)
        except InvalidArgument:
            """
            This happens when the function expects an object but gets
            a list of objects because of a different number of workers
            and parameters. If the function cannot handle a list of objects,
            then the library calls the function in series for each object.
            """
            result = None
        return result

class DistributedFunction():
    def __init__(self,  fun = None,
                        endaftercall = True,
                        machines = [],
                        gpu_policy = 'no_gpu',
                        max_cpu = None,
                        max_gpu = None,
                        named_pipe = None,
                        port = None,
                        authkey = 'distributedfunction',
                        accept_lists = False, # Set to True if the provided function handles a list as a parameter
                        **shared_data):
        
        if fun is None:
            raise ValueError('The function must be provided')
        
        self.endaftercall = endaftercall
        if shared_data is None:
            shared_data = dict([])
        shared_data['_fun'] = fun
        self.manager = ClusterManager(DistributedWorker, 
                                      shared_data = shared_data,
                                      machines = machines,
                                      gpu_policy = gpu_policy,
                                      own_max_cpu = max_cpu,
                                      own_max_gpu = max_gpu,
                                      named_pipe = named_pipe,
                                      port = port,
                                      authkey = authkey) 
        self.numprocesses = self.manager.total_processes
        self.accept_lists = accept_lists
        
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
#        logging.info("Using %d %s%s..." % (self.numprocesses, cores, b))

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

def distributedslave(max_cpu = None, max_gpu = None, port = None,
                      named_pipe = None, authkey = 'distributedfunction'):
    cluster_worker_script(DistributedWorker,
                          max_gpu=max_gpu, max_cpu=max_cpu, port=port,
                          named_pipe=named_pipe, authkey=authkey)
