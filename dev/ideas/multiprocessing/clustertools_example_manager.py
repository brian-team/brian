from numpy import *
from clustertools import *

class work_class(object):
    def __init__(self, shared_data):
        self.x = shared_data['x']
    def process(self, job):
        return sum(self.x)*job

if __name__=='__main__':
    shared_data = {'x':ones(100)}
    manager = ClusterManager(work_class, shared_data, machines=['localhost'])
    results = manager.process_jobs(arange(manager.total_processes)+1)
    manager.finished()
    print results
    