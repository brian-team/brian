from numpy import exp

def fun(args, shared_data, local_data, use_gpu):
    x = args['x']
    y = args['y']
    return exp(-(x**2+y**2))

if __name__ == '__main__':
    from distopt import *
    optparams = dict(x = [-10.,10.], y = [-10.,10.])
    results = optimize(fun, optparams)
    print_results(results)
    