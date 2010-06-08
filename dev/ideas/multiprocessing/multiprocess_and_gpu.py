'''
Sample code showing running multiple processes including multiple GPUs

See Brian/examples/multiprocessing/* for detailed examples with comments.

Notes for Cyrille:

This file works whether or not a GPU is present. If there is a GPU then it
will launch num_gpus processes, otherwise it will launch num_cpus processes.
Things to know:

* Launch one process for each CPU/GPU
* Have a function f and a set of arguments x
* Each value f(x) is computed in a separate process
* The arguments x need to be picklable. Most Brian objects are not, so you'll
  need to pass parameters etc. - enough parameters that you can reconstruct
  the ModelFitting objects in each separate process
* If using the GPU, you need to pass a copy of the process number to each
  process so that it knows which GPU to use
'''

import multiprocessing
try:
    import pycuda.autoinit as autoinit
    import pycuda.driver as drv
    def set_gpu_device(n):
        '''
        This function makes pycuda use GPU number n in the system.
        '''
        global _gpu_context # we make this a global variable so that a
                            # reference is kept to it and the context is not
                            # deallocated
        autoinit.context.pop() # delete the old context before making a new one
        _gpu_context = drv.Device(n).make_context()
    use_gpu = True
except ImportError:
    use_gpu = False

# Note that func actually takes only one argument, that argument is a tuple
# of arguments (which is why there are double parentheses). The reason is that
# pool.map only works for functions of one argument. Note also that the arguments
# have to be picklable and not all Brian objects are, however all numpy objects
# should be.
def func((process_n, x)):
    # if we are using the GPU then there should be one process for each
    # GPU, and process_n should use GPU number process_n.
    if use_gpu:
        set_gpu_device(process_n)
    # just a trivial computation to give the idea
    return x ** 2

if __name__ == '__main__':
    # If we have GPUs we use them, otherwise we use CPUs
    if use_gpu:
        numprocesses = drv.Device.count() # number of GPUs present in the system
        print 'Using %d GPUs' % numprocesses
    else:
        numprocesses = multiprocessing.cpu_count() # number of CPUs in the system
        print 'Using %d CPUs' % numprocesses
    pool = multiprocessing.Pool(processes=numprocesses)
    # args consists of pairs (process_n, x) where process_n is the number of the
    # process, which is used to initialiase the correct GPU if one is present,
    # and x is the argument to the function (see notes in module docstring
    # about this)
    args = zip(range(numprocesses), range(2, numprocesses + 2))
    results = pool.map(func, args) # launches multiple processes
    for (pid, arg), res in zip(args, results):
        print 'Process', pid, 'argument', arg, 'result', res
