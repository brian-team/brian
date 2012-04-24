from brian import *
from brian.experimental.codegen2 import *
import pycuda
from pycuda import gpuarray
from pycuda import compiler
from pycuda import scan
from pycuda import driver        

if __name__=='__main__':
    
    N = 10000
    p = 100*Hz*defaultclock.dt#0.001
    index_dtype = int64
    index_typestr = 'int64_t'
    
    x = p-rand(N)
    truexc, = (x>0).nonzero()
    gx = gpuarray.to_gpu(x)
    
    #print 'numspiked =', len(truexc)
    
    threshold_code = '''
    __global__ void threshold(double *x, INT *spiked, INT N)
    {
        const int thread = threadIdx.x+blockIdx.x*blockDim.x;
        if(thread>=N) return;
        spiked[thread] = (INT)(x[thread]>0.0);
    }
    '''.replace('INT', index_typestr)
    
    threshold_module = compiler.SourceModule(threshold_code)
    threshold_func = threshold_module.get_function('threshold')
    
    spiked = array(hstack((x>0, False)), dtype=index_dtype)
    gspiked = gpuarray.zeros(N+1, dtype=index_dtype)
    
    block, grid = compute_block_grid(1024, N)
    
    threshold_func(gx, gspiked, index_dtype(N), block=block, grid=grid)

    if not equal(spiked-gspiked.get(), 0).all():
        print 'gspiked incorrect!'
        print spiked
        print gspiked
        exit() 
    
    threshold_func(gx, gspiked, index_dtype(N), block=block, grid=grid)
    scancomp = GPUScanCompactor(N, index_dtype=index_dtype)
    xc = scancomp(gspiked)
    correct = (len(xc)==len(truexc) and equal(xc, truexc).all())
    print 'GPUScanCompactor correct', correct
    
    threshold_func(gx, gspiked, index_dtype(N), block=block, grid=grid)
    atomcomp = GPUAtomicCompactor(N, index_dtype=index_dtype)
    xc = atomcomp(gspiked)
    correct = (len(xc)==len(truexc) and equal(xc, truexc).all())
    print 'GPUAtomicCompactor correct', correct
    
    #exit()

    repeats = 1000
    import time
    start = time.time()    
    for i in xrange(repeats):
        threshold_func(gx, gspiked, index_dtype(N), block=block, grid=grid)
    end = time.time()
    threshold_time = end-start
    
    start = time.time()    
    for i in xrange(repeats):
        threshold_func(gx, gspiked, index_dtype(N), block=block, grid=grid)
        scancomp.get_gpu(gspiked)
    end = time.time()
    gpu_only_time = end-start

    start = time.time()    
    for i in xrange(repeats):
        threshold_func(gx, gspiked, index_dtype(N), block=block, grid=grid)
        scancomp(gspiked)
    end = time.time()
    comp_time = end-start

    start = time.time()    
    for i in xrange(repeats):
        threshold_func(gx, gspiked, index_dtype(N), block=block, grid=grid)
        spiked = gspiked.get()[:-1]
        spiked.nonzero()
    end = time.time()
    cpu_time = end-start

    start = time.time() 
    for i in xrange(repeats):
        threshold_func(gx, gspiked, index_dtype(N), block=block, grid=grid)
        atomcomp(gspiked)
    end = time.time()
    atom_time = end-start

    uatomcomp = GPUAtomicCompactor(N, sorted=False, index_dtype=index_dtype)
    start = time.time() 
    for i in xrange(repeats):
        threshold_func(gx, gspiked, index_dtype(N), block=block, grid=grid)
        atomcomp(gspiked)
    end = time.time()
    atom_unsorted_time = end-start
    
    print
    print 'PROFILING N =', N, 'p =', p
    print
    print 'Time using CPU after download:', (cpu_time-threshold_time)/repeats*second
    print 'Time with scan compactor (GPU only):', (gpu_only_time-threshold_time)/repeats*second
    print 'Time with scan compactor (inc download):', (comp_time-threshold_time)/repeats*second
    print 'Time with atomic compactor (inc CPU sort):', (atom_time-threshold_time)/repeats*second
    print 'Time with atomic compactor (unsorted):', (atom_unsorted_time-threshold_time)/repeats*second
    