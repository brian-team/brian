from numpy import *
from clustertools import *

class work_class(object):
    def __init__(self, shared_data, use_gpu):
        self.x = shared_data['x']
    def process(self, job):
        return sum(self.x)*job

if __name__=='__main__':
    shared_data = {'x':ones(100)}
    manager = ClusterManager(work_class, shared_data,
#                             machines=['129.199.80.199'],
#                             port=80,
#                             machines=['localhost'],
                             machines=[r'\\Romain-PC\pipe\ModelFitting'],
#                             machines=[r'\\Astrance\pipe\ModelFitting'],
#                             gpu_policy='require_all',
                             own_max_gpu=0,
                             own_max_cpu=2,
                             port=None,
                             named_pipe=True,#r'\\Astrance\pipe\ModelFitting'
                             )
    results = manager.process_jobs(arange(manager.total_processes)+1)
    manager.finished()
    print results
    