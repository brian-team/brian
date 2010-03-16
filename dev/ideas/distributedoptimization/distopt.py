from clustertools import *
from optmanager import *
from numpy import exp

def optimize(   fun, 
                optparams,
                shared_data = None,
                local_data = None,
                group_size = None,
                group_count = None,
                machines = [],
                gpu_policy = 'no_gpu',
                max_cpu = None,
                max_gpu = None,
                named_pipe = None,
                port = None,
                iterations = None,
                optinfo = None,
                returninfo = False
                ):
        
        if group_size is None:
            group_size = 1
        
        if group_count is None:
            group_count = 1
            
        if iterations is None:
            iterations = 1
        
        if shared_data is None:
            shared_data = dict([])
        
        if optinfo is None:
            optinfo = dict([])
        
        if optparams is None:
            raise Exception('optparams must be specified.')
            
        shared_data['group_size'] = group_size
        shared_data['group_count'] = group_count
        shared_data['returninfo'] = returninfo
        shared_data['optparams'] = optparams
        shared_data['optinfo'] = optinfo
        optinfo['iterations'] = iterations
        shared_data['fun'] = fun
        
        clusterinfo = dict(machines = machines,
                            gpu_policy = gpu_policy,
                            max_cpu = max_cpu,
                            max_gpu = max_gpu,
                            named_pipe = named_pipe,
                            port = port)
    
        fm = OptManager(shared_data, local_data, clusterinfo, optinfo)
        fm.run()
        results, fitinfo = fm.get_results()
        
        return results

