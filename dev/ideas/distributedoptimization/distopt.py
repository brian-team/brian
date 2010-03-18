from clustertools import *
from optmanager import *
from optworker import *
from numpy import exp, ndarray, floor, log10

__all__ = ['optimize', 'print_results', 'optworker']

def optimize(   fun, 
                optparams,
                shared_data = None,
                local_data = None,
                group_size = None,
                group_count = None,
                iterations = None,
                optinfo = None,
                machines = [],
                gpu_policy = 'no_gpu',
                max_cpu = None,
                max_gpu = None,
                named_pipe = None,
                port = None,
                returninfo = False,
                verbose = False,
                ):
        
        if group_size is None:
            group_size = 100
        
        if group_count is None:
            group_count = 1
        
        # Checks local_data
        if local_data is not None:
            for key, val in local_data.iteritems():
                if type(val) == list:
                    if len(val) != group_count:
                        raise Exception('Each local_data value must have as many elements as group_count')
                if type(val) == ndarray:
                    if val.shape[-1] != group_count:
                        raise Exception('The last dimension of each local_data array must be equal to group_count')
            
        if iterations is None:
            iterations = 10
        
        if shared_data is None:
            shared_data = dict([])
        
        if optinfo is None:
            optinfo = dict([])
        
        if optparams is None:
            raise Exception('optparams must be specified.')
            
        shared_data['_fun'] = fun
        shared_data['_group_size'] = group_size
        shared_data['_group_count'] = group_count
        shared_data['_returninfo'] = returninfo
        shared_data['_optparams'] = optparams
        shared_data['_optinfo'] = optinfo
        shared_data['_verbose'] = verbose
        
        # Adds iterations to optinfo
        optinfo['iterations'] = iterations
        
        # Creates the clusterinfo object
        clusterinfo = dict(machines = machines,
                            gpu_policy = gpu_policy,
                            max_cpu = max_cpu,
                            max_gpu = max_gpu,
                            named_pipe = named_pipe,
                            port = port)
    
        fm = OptManager(shared_data, local_data, clusterinfo, optinfo)
        fm.run()
        
        if returninfo:
            results, fitinfo = fm.get_results()
            return results, fitinfo
        else:
            results = fm.get_results()
            return results

def optworker(max_cpu = None, max_gpu = None, port = None,
                      named_pipe = None):
    cluster_worker_script(OptWorker,
                          max_gpu=max_gpu, max_cpu=max_cpu, port=port,
                          named_pipe=named_pipe, authkey='distopt')

def print_quantity(x, precision=4):
    if x == 0.0:
        u = 0
    else:
        u = int(3*floor((log10(abs(x))+1)/3))
    s = ('%.'+str(precision)+'f') % float(x/(10**u))
    if u is not 0:
        su = "e"+str(u)
    else:
        su = ''
    return s+su

def print_results(results, precision=4, colwidth=16):
    group_count = len(results['fitness'])
    
    print "RESULTS"
    print '-'*colwidth*(group_count+1)
    
    print ' '*colwidth,
    for i in xrange(group_count):
        s = 'Group %d' % i
        spaces = ' '*(colwidth-len(s))
        print s+spaces,
    print
    
    def print_row(name, values):
        spaces = ' '*(colwidth-len(name))
        print name+spaces,
        for value in values:
            s = print_quantity(value)
            spaces = ' '*(colwidth-len(s))
            print s+spaces,
        print
    
    keys = results.keys()
    keys.sort()
    for key in keys:
        val = results[key]
        if key != 'fitness':
            print_row(key, val)
    
    print_row('fitness', results['fitness'])
        