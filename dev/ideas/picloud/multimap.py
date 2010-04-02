import cloud

__all__ = ['multimap', 'retrieve']

api_keys = ['1307',
            '1308']

api_secretkeys = ['9106bed9b15b00197df2734102a66a9ce5698f1d',
                  '69d0919b47fab35e959bd7762c163bd4826a393c']

def multimap(fun, args):
    n = len(args)/len(api_keys)
    
    # jids[i] contains the job indices for account i
    jids = [None for _ in xrange(len(api_keys))]
    
    # Launches the jobs
    k = 0
    for i in xrange(len(api_keys)):
        api_key = api_keys[i]
        api_secretkey = api_secretkeys[i]
        
        args_tmp = args[k:k+n]
        if len(args_tmp)>0:
            print "Launching %d jobs with account %d..." % (len(args_tmp), i+1)
            cloud.setkey(api_key=api_key, api_secretkey=api_secretkey)
            jids[i] = cloud.map(fun, args_tmp)
            print jids[i]
            k += n
    return jids

def retrieve(jids):
    results = []
    # Retrieves the results
    for i in xrange(len(api_keys)):
        api_key = api_keys[i]
        api_secretkey = api_secretkeys[i]
        cloud.setkey(api_key=api_key, api_secretkey=api_secretkey)
        print "Retrieving results for account %d..." % (i+1)
        results.extend(cloud.result(jids[i]))
        
    return results

