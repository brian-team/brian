import cloud

__all__ = ['multimap', 'retrieve']

api_keys = ['1307',# Cyrille
            '1308',
            '1360',# Romain
            '1366']

api_secretkeys = ['9106bed9b15b00197df2734102a66a9ce5698f1d',
                  '69d0919b47fab35e959bd7762c163bd4826a393c',
                  '6400d6d25914f3bd01580ab44c7f7ea06bb77908',
                  '7996f5335cf66e54f56a7c5000c28e1c96bdd001']

def multimap(fun, args, naccounts = None):
    
    if naccounts is None:
        naccounts = len(api_keys)
    
    n = len(args)/naccounts
    
    # jids[i] contains the job indices for account i
    jids = [None for _ in xrange(naccounts)]
    
    # Launches the jobs
    k = 0
    for i in xrange(naccounts):
        api_key = api_keys[i]
        api_secretkey = api_secretkeys[i]
        
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

