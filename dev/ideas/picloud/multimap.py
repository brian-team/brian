import cloud
from keys import *
from numpy import array, cumsum, nonzero


naccounts = len(api_keys)
max_parallel = sum(array(parallelism[:naccounts]))
print "Current parallelism level : %d" % max_parallel

__all__ = ['multimap', 'retrieve', 'status', 'naccounts', 'max_parallel']

def multimap(fun, args, naccounts = None):
    if naccounts is None:
        naccounts = len(api_keys)
        
    max_parallel = sum(array(parallelism[:naccounts]))
    if len(args) <= max_parallel:
        naccounts = nonzero(cumsum(parallelism)-len(args)>=0)[0][0]+1
        size = parallelism[:naccounts]
    else:
        size = [len(args)/naccounts for _ in xrange(naccounts)]
    if naccounts>1:
        size[-1] = len(args)-sum(array(size[:-1]))
    else:
        size[0] = len(args)
    
    # jids[i] contains the job indices for account i
    jids = [None for _ in xrange(naccounts)]
    
    # Launches the jobs
    k = 0
    for i in xrange(naccounts):
        api_key = api_keys[i]
        api_secretkey = api_secretkeys[i]
        n = size[i]
        
        args_tmp = args[k:k+n]
        if len(args_tmp)>0:
            print "Launching %d jobs with account %d..." % (len(args_tmp), i+1)
            cloud.setkey(api_key=api_key, api_secretkey=api_secretkey)
            jids[i] = cloud.map(fun, args_tmp, _high_cpu=True)
            print "    Jobs:", jids[i]
            k += n
    return jids

def retrieve(jids):
    naccounts = len(jids)
    results = []
    # Retrieves the results
    for i in xrange(naccounts):
        api_key = api_keys[i]
        api_secretkey = api_secretkeys[i]
        cloud.setkey(api_key=api_key, api_secretkey=api_secretkey)
        print "Retrieving results for account %d..." % (i+1)
        results.extend(cloud.result(jids[i]))
        
    return results

def status(jids):
    naccounts = len(jids)
    status = []
    # Retrieves the results
    for i in xrange(naccounts):
        api_key = api_keys[i]
        api_secretkey = api_secretkeys[i]
        cloud.setkey(api_key=api_key, api_secretkey=api_secretkey)
        print "Retrieving status for account %d..." % (i+1)
        status.extend(cloud.status(jids[i]))
        
    return status