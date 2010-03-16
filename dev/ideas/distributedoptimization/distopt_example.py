from clustertools import *
from numpy import exp

def fun(args, shared_data, local_data, use_gpu):
    a = args['a']
    b = args['b']
    return exp(-(a**2+b**2)/.5)

if __name__ == '__main__':
    import distopt
    optparams = dict(a=[-1.,1.],b=[-1.,1.])
    result = distopt.optimize(fun, optparams, max_cpu=1) 
    
    print results
    