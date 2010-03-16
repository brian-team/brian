from numpy import split, concatenate, ndarray, cumsum, ndim, isscalar
from clustertools import *
import sys
#import logging

all = ['DistributedWorker', 'DistributedFunction', 'distribute']

class DistributedWorker:
    def __init__(self, shared_data, use_gpu):
        self.fun = shared_data['fun']
    
    def process(self, job):
        result = self.fun(job)
#        print job
#        sys.stdout.flush()
        return result

class DistributedFunction():
    def __init__(self, fun = None, endaftercall = True, 
                    machines = [],
                    gpu_policy = 'prefer_gpu',
                    own_max_cpu = None,
                    own_max_gpu = None,
                    named_pipe = None,
                    port = None,
                    authkey = None):
        
        if fun is None:
            raise ValueError('The function must be provided')
        
        self.endaftercall = endaftercall
            
        self.manager = ClusterManager(DistributedWorker, 
                                      shared_data = dict(fun = fun),
                                      machines = machines,
                                      gpu_policy = gpu_policy,
                                      own_max_cpu = own_max_cpu,
                                      own_max_gpu = own_max_gpu,
                                      named_pipe = named_pipe,
                                      port = port,
                                      authkey = authkey)
        self.numprocesses = self.manager.total_processes
        
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
        if x is None:
            jobs = [None for _ in xrange(self.numprocesses)]
        elif isinstance(x, list):
            n = len(x)
            if n == self.numprocesses:
                jobs = x
            else:
                worker_size, bins = self.divide(n)
                jobs = [x[bins[i]:bins[i+1]] for i in xrange(self.numprocesses)]
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
        return jobs

    def __call__(self, x = None):
        jobs = self.prepare_jobs(x)
        results = self.manager.process_jobs(jobs)
        
        if isinstance(x, ndarray):
            results = concatenate(results, axis=-1)
        elif isinstance(x, list):
            results2 = []
            for r in results:
                if isscalar(r):
                    results2.append(r)
                else:
                    results2.extend(r)
            results = results2
        
        if self.endaftercall:
            self.end()
        return results
    
    def __del__(self):
        self.end()
        
    def end(self):
        self.manager.finished()

def distribute(fun, endaftercall = True, 
                    machines = [],
                    gpu_policy = 'prefer_gpu',
                    max_cpu = None,
                    max_gpu = None,
                    named_pipe = None,
                    port = None,
                    authkey = 'distributedfunction'):
    dfun = DistributedFunction(fun,
                                machines = machines,
                                gpu_policy = gpu_policy,
                                own_max_cpu = max_cpu,
                                own_max_gpu = max_gpu,
                                named_pipe = named_pipe,
                                port = port,
                                authkey = authkey)
    return dfun

