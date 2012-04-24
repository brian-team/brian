from brian import *
from management import *
from importpycuda import *

__all__ = ['GPUCompactor',
           'GPUScanCompactor', 'GPUAtomicCompactor']

class GPUScanCompactor(object):
    def __init__(self, N, maxblocksize=None, index_dtype=int32):
        from ..statements import c_data_type
        if maxblocksize is None:
            maxblocksize = pycuda.autoinit.device.max_threads_per_block
        self.index_dtype = index_dtype
        self.index_typestr = index_typestr = c_data_type(index_dtype)
        self.nspiked_arr = zeros(1, dtype=index_dtype)
        self.cumsum_kernel = scan.ExclusiveScanKernel(index_dtype, 'a+b', 0)
        scatter_code = '''
        __global__ void scatter(INT *out, INT *indices, INT N)
        {
            const int thread = threadIdx.x+blockIdx.x*blockDim.x;
            if(thread>=N) return;
            if(indices[thread]!=indices[thread+1])
                out[indices[thread]] = thread;
        }
        '''.replace('INT', index_typestr)
        scatter_module = compiler.SourceModule(scatter_code)
        self.scatter_func = scatter_module.get_function('scatter')
        self.block, self.grid = compute_block_grid(maxblocksize, N)
        self.N = index_dtype(N)
        self.scatter_func.prepare((intp, intp, index_dtype), self.block)
        self.gpu_compacted = gpuarray.zeros(N, index_dtype)
        self.compacted = zeros(N, index_dtype)
    def get_gpu(self, cond):
        N = self.N
        if len(cond)!=N+1:
            raise ValueError("Should have len(cond)=N+1")
        self.cumsum_kernel(cond)
        driver.memcpy_dtoh(self.nspiked_arr, int(int(cond.gpudata)+N*self.index_dtype(0).nbytes))
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
    def __init__(self, N, maxblocksize=None, sorted=True, index_dtype=int32):
        from ..statements import c_data_type
        if maxblocksize is None:
            maxblocksize = pycuda.autoinit.device.max_threads_per_block
        self.index_dtype = index_dtype
        self.index_typestr = index_typestr = c_data_type(index_dtype)
        self.block, self.grid = compute_block_grid(maxblocksize, N)
        self.N = index_dtype(N)
        self.compact_code = '''
        __global__ void compact(INT *out, INT *cond, INT N,
                                unsigned int *counter)
        {
            const int thread = threadIdx.x+blockIdx.x*blockDim.x;
            if(thread>=N) return;
            if(cond[thread])
                out[atomicAdd(counter, 1)] = thread;
        }
        '''.replace('INT', index_typestr)
        self.module = compiler.SourceModule(self.compact_code)
        self.gpu_compact = self.module.get_function('compact')
        self.gpu_compact.prepare((intp, intp, index_dtype, intp), self.block)
        self.gpu_compacted = gpuarray.zeros(N, index_dtype)
        self.compacted = zeros(N, index_dtype)
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

# TODO: could make this an option, or switch based on runtime profiling, but
# in most realistic cases, atomic seems to be slightly faster than scan
# where realistic means firing rates lower than 100 Hz and N<1M
GPUCompactor = GPUAtomicCompactor
