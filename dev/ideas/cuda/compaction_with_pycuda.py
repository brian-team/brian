from brian import *
from brian.experimental.codegen2.gpu import *
import pycuda
from pycuda import gpuarray
from pycuda import compiler
from pycuda import scan
from pycuda import driver

__all__ = ['GPUScanCompactor', 'GPUAtomicCompactor']

class GPUScanCompactor(object):
    def __init__(self, N, maxblocksize=None):
        if maxblocksize is None:
            maxblocksize = pycuda.autoinit.device.max_threads_per_block
        self.nspiked_arr = zeros(1, dtype=int64)
        self.cumsum_kernel = scan.ExclusiveScanKernel(int64, 'a+b', 0)
        scatter_code = '''
        __global__ void scatter(int64_t *out, int64_t *indices, int64_t N)
        {
            const int thread = threadIdx.x+blockIdx.x*blockDim.x;
            if(thread>=N) return;
            if(indices[thread]!=indices[thread+1])
                out[indices[thread]] = thread;
        }
        '''
        scatter_module = compiler.SourceModule(scatter_code)
        self.scatter_func = scatter_module.get_function('scatter')
        self.block, self.grid = compute_block_grid(maxblocksize, N)
        self.N = int64(N)
        self.scatter_func.prepare((intp, intp, int64), self.block)
        self.gpu_compacted = gpuarray.zeros(N, int64)
        self.compacted = zeros(N, int64)
    def get_gpu(self, cond):
        N = self.N
        if len(cond)!=N+1:
            raise ValueError("Should have len(cond)=N+1")
        self.cumsum_kernel(cond)
        driver.memcpy_dtoh(self.nspiked_arr, int(int(cond.gpudata)+N*int64(0).nbytes))
        nspiked = self.nspiked_arr[0]
        self.scatter_func.prepared_call(self.grid,
                                        intp(self.gpu_compacted.gpudata),
                                        intp(cond.gpudata),
                                        N)
        return self.gpu_compacted, nspiked
    def __call__(self, cond):
        gpucomp, nspiked = self.get_gpu(cond)
        compacted = self.compacted[:nspiked]        
        driver.memcpy_dtoh(compacted, gpucomp.gpudata)
        return compacted


class GPUAtomicCompactor(object):
    def __init__(self, N, maxblocksize=None, sorted=True):
        if maxblocksize is None:
            maxblocksize = pycuda.autoinit.device.max_threads_per_block
        self.block, self.grid = compute_block_grid(maxblocksize, N)
        self.N = int64(N)
        self.compact_code = '''
        __global__ void compact(int64_t *out, int64_t *cond, int N,
                                unsigned int *counter)
        {
            const int thread = threadIdx.x+blockIdx.x*blockDim.x;
            if(thread>=N) return;
            if(cond[thread])
                out[atomicAdd(counter, 1)] = thread;
        }
        '''
        self.module = compiler.SourceModule(self.compact_code)
        self.gpu_compact = self.module.get_function('compact')
        self.gpu_compact.prepare((intp, intp, int64, intp), self.block)
        self.gpu_compacted = gpuarray.zeros(N, int64)
        self.compacted = zeros(N, int64)
        self.counter = gpuarray.zeros(1, uint32)
        self.sorted = sorted
    def __call__(self, cond):
        N = self.N
        if len(cond)!=N+1:
            raise ValueError("Should have len(cond)=N+1")
        self.counter.set(array([0], dtype=uint32))
        self.gpu_compact.prepared_call(self.grid,
                                       intp(self.gpu_compacted.gpudata),
                                       intp(cond.gpudata),
                                       self.N,
                                       intp(self.counter.gpudata))
        nspiked = self.counter.get()[0]
        #print nspiked
        compacted = self.compacted[:nspiked]        
        driver.memcpy_dtoh(compacted, self.gpu_compacted.gpudata)
        if self.sorted:
            compacted.sort()
        return compacted
        

if __name__=='__main__':
    
    N = 1000000
    p = 0.001
    
    x = p-rand(N)
    truexc, = (x>0).nonzero()
    gx = gpuarray.to_gpu(x)
    
    #print 'numspiked =', len(truexc)
    
    threshold_code = '''
    __global__ void threshold(double *x, int64_t *spiked, int64_t N)
    {
        const int thread = threadIdx.x+blockIdx.x*blockDim.x;
        if(thread>=N) return;
        spiked[thread] = (int64_t)(x[thread]>0.0);
    }
    '''
    
    threshold_module = compiler.SourceModule(threshold_code)
    threshold_func = threshold_module.get_function('threshold')
    
    spiked = array(hstack((x>0, False)), dtype=int)
    gspiked = gpuarray.zeros(N+1, dtype=int64)
    
    block, grid = compute_block_grid(1024, N)
    
    threshold_func(gx, gspiked, int64(N), block=block, grid=grid)

    if not equal(spiked-gspiked.get(), 0).all():
        print 'gspiked incorrect!'
        print spiked
        print gspiked
        exit() 
    
    threshold_func(gx, gspiked, int64(N), block=block, grid=grid)
    scancomp = GPUScanCompactor(N)
    xc = scancomp(gspiked)
    correct = (len(xc)==len(truexc) and equal(xc, truexc).all())
    print 'GPUScanCompactor correct', correct
    
    threshold_func(gx, gspiked, int64(N), block=block, grid=grid)
    atomcomp = GPUAtomicCompactor(N)
    xc = atomcomp(gspiked)
    correct = (len(xc)==len(truexc) and equal(xc, truexc).all())
    print 'GPUAtomicCompactor correct', correct
    
    #exit()

    repeats = 100
    import time
    start = time.time()    
    for i in xrange(repeats):
        threshold_func(gx, gspiked, int64(N), block=block, grid=grid)
    end = time.time()
    threshold_time = end-start
    
    start = time.time()    
    for i in xrange(repeats):
        threshold_func(gx, gspiked, int64(N), block=block, grid=grid)
        scancomp.get_gpu(gspiked)
    end = time.time()
    gpu_only_time = end-start

    start = time.time()    
    for i in xrange(repeats):
        threshold_func(gx, gspiked, int64(N), block=block, grid=grid)
        scancomp(gspiked)
    end = time.time()
    comp_time = end-start

    start = time.time()    
    for i in xrange(repeats):
        threshold_func(gx, gspiked, int64(N), block=block, grid=grid)
        spiked = gspiked.get()[:-1]
        spiked.nonzero()
    end = time.time()
    cpu_time = end-start

    start = time.time() 
    for i in xrange(repeats):
        threshold_func(gx, gspiked, int64(N), block=block, grid=grid)
        atomcomp(gspiked)
    end = time.time()
    atom_time = end-start

    uatomcomp = GPUAtomicCompactor(N, sorted=False)
    start = time.time() 
    for i in xrange(repeats):
        threshold_func(gx, gspiked, int64(N), block=block, grid=grid)
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
    