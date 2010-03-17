from numpy import exp

def fun(args, shared_data, local_data, use_gpu):
    a = args['a']
    b = args['b']
    try:
        a0 = local_data['a0']
        b0 = local_data['b0']
    except:
        a0 = b0 = 0
    sigma = shared_data['sigma']
    return exp(-((a-a0)**2+(b-b0)**2)/(2*sigma*2))

if __name__ == '__main__':
    from distopt import *
    optparams = dict(a = [-10.,10.], b = [-10.,10.])
    shared_data = dict(sigma = 1.0)
    group_size = 2000
    local_data = dict(a0 = [1.0, 2.0], b0 = [3.0, 4.0])
    results = optimize(fun, optparams, shared_data, local_data,
                       group_size = 37, group_count = 2,
                       iterations = 10, verbose = True)
    print_results(results)
    