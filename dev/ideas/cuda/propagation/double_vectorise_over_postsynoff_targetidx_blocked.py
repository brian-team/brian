'''
Notes:

This algorithm is based on Issam's forDan/283_end.cu.

The idea (from Romain) is to do basically the same thing as in
vectorise_over_postsynaptic_offset but we do a double vectorisation. In the
first pass, the vectorisation is over the postsynaptic offset, but this time
we accumulate into a shared memory buffer without atomics (but this SHOULD
be done atomically right? just in shared memory atomics which is nicer). After,
we first sync the threads, then we propagate the shared memory buffer into
global memory.

Note that in Issam's version there are two potential bugs:
1. shared memory propagation is done without atomics, but I think conflicts are
   still possible.
2. it only works if the number of postsynaptic neurons per presynaptic neuron
   is always less than the shared memory size.

Performance:
- everything is coalesced except the shared memory atomics, but these are not
too costly. On the other hand, we potentially waste a lot of time
propagating unnecessarily from shared to global memory. This might be fixed by
having warp-sized boolean flags to say whether or not a propagation is needed.
'''

from brian import *
from brian.experimental.codegen2 import *
import numpy
import time
import numpy.random as nrandom
import random as prandom
try:
    import pycuda
except ImportError:
    pycuda = None    

__all__ = ['GPUConnection']

class GPUConnection(Connection):
    '''
    Only works with sparse at the moment, no modulation, no delays
    '''
    def __init__(self, *args, **kwds):
        self.use_atomic = kwds.pop('use_atomic', False)
        self.masked_write = kwds.pop('masked_write', True)
        super(GPUConnection, self).__init__(*args, **kwds)
    def gpu_prepare(self):
        self._gpu_prepared = True
        # find target variable on GPU
        self.gpuman = gpuman = self.target._state_updater.language.gpu_man
        self.memman = memman = gpuman.mem_man
        self.gpu_target_var = self.memman.device['_arr_'+self.state]
        blocksize = 1024
        # upload relevant data
        W = self.W
        self.gpu_alldata = pycuda.gpuarray.to_gpu(W.alldata)
        self.gpu_allj = pycuda.gpuarray.to_gpu(W.allj)
        self.gpu_spikes = pycuda.gpuarray.empty(len(self.source), dtype=int)
        # define new rowind and numsynapses
        numblocks = int(ceil(float(len(self.target))/blocksize))
        rowind = zeros((numblocks+1)*len(self.source), dtype=int)
        maxnumsynapses = 0
        for i, row in enumerate(W.rows):
            rowoff = searchsorted(row.ind/blocksize, arange(numblocks+1))+W.rowind[i]
            rowind[i*(numblocks+1):(i+1)*(numblocks+1)] = rowoff
            maxnumsynapses = max(maxnumsynapses, amax(diff(rowoff)))
#            print rowoff, len(row.ind)
#            print row.ind
#            for j in xrange(len(rowoff)-1):
#                print j, W.allj[rowoff[j]:rowoff[j+1]]
#            if i==3:
#                exit()
        self.gpu_rowind = pycuda.gpuarray.to_gpu(rowind)    
        # define propagation kernel
        self.gpu_code = '''
        // From CUDA manual
        __device__ double atomicAdd(double* address, double val)
        {
            unsigned long long int* address_as_ull = (unsigned long long int*)address;
            unsigned long long int old = *address_as_ull, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
            } while (assumed != old);
            return __longlong_as_double(old);
        }
        // based on Issam's code
        __global__ void propagate(double *v, double *alldata, int64_t *rowind,
                                  int64_t *allj,
                                  int64_t target_offset,
                                  int64_t maxnumsynapses,
                                  int64_t *spikes, int64_t numspikes)
        {
            //return;
            __shared__ double stage[%BLOCKSIZE%];
            // zero stage memory
            stage[threadIdx.x] = 0.0;
            __syncthreads();
            // propagate to stage
            for(int spikes_index=0; spikes_index<numspikes; spikes_index++)
            {
                const int spike = spikes[spikes_index];
                const int blockoffset = spike*(%NUMBLOCKS%+1)+blockIdx.x;
                const int dataoffset = rowind[blockoffset]+threadIdx.x;
                if(dataoffset<rowind[blockoffset+1])
                {
                    const int target = allj[dataoffset]% (%BLOCKSIZE%); // coalesced
                    const double weight = alldata[dataoffset]; // coalesced
                    %PROPAGATE%
                    //stage[target] += weight; // uncoalesced, incorrect, but no atomics
                    //atomicAdd(stage+target, weight); // uncoalesced, correct
                }
            }
            __syncthreads();
            // propagate to target
            // TODO: could do something clever on devices with warp vote functions?
            %MASKED_WRITE%
            //if(stage[threadIdx.x]==0.0)
            //    return;
            const int target_neuron = blockIdx.x * blockDim.x + threadIdx.x;
            v[target_offset+target_neuron] += stage[threadIdx.x];
        }
        '''
        self.gpu_code = self.gpu_code.replace('%NUMBLOCKS%', str(numblocks))
        self.gpu_code = self.gpu_code.replace('%BLOCKSIZE%', str(blocksize))
        if self.use_atomic:
            self.gpu_code = self.gpu_code.replace('%PROPAGATE%',
                    'atomicAdd(stage+target, weight);')
        else:
            self.gpu_code = self.gpu_code.replace('%PROPAGATE%',
                    'stage[target] += weight;')
        if self.masked_write:
            self.gpu_code = self.gpu_code.replace('%MASKED_WRITE%',
                      'if(stage[threadIdx.x]==0.0) return;')
        else:
            self.gpu_code = self.gpu_code.replace('%MASKED_WRITE%', '')
        log_info('brian.GPUConnection', 'code:\n'+self.gpu_code)
        self.gpu_module = pycuda.compiler.SourceModule(self.gpu_code)
        self.gpu_func = self.gpu_module.get_function('propagate')
        log_info('brian.GPUConnection',  'local size: '+str(self.gpu_func.local_size_bytes))
        log_info('brian.GPUConnection',  'shared size: '+str(self.gpu_func.shared_size_bytes))
        log_info('brian.GPUConnection',  'num regs: '+str(self.gpu_func.num_regs))
        self.block = (blocksize, 1, 1)
        self.grid = (numblocks, 1)
        log_info('brian.GPUConnection', 'block='+str(self.block)+', grid='+str(self.grid))
        self.gpu_func.prepare(
                (numpy.intp, numpy.intp, numpy.intp, numpy.intp,
                 numpy.int64, numpy.int64, numpy.intp, numpy.int64),
                self.block)
        self.gpu_args = (numpy.intp(self.gpu_target_var.gpudata),
                         numpy.intp(self.gpu_alldata.gpudata),
                         numpy.intp(self.gpu_rowind.gpudata),
                         numpy.intp(self.gpu_allj.gpudata),
                         numpy.int64(self.target._origin),
                         numpy.int64(maxnumsynapses),
                         numpy.intp(self.gpu_spikes.gpudata),
                         )
        
    def propagate(self, spikes):
        if not hasattr(self, '_gpu_prepared'):
            self.gpu_prepare()
        if len(spikes)==0:
            return
        if not isinstance(spikes, numpy.ndarray):
            spikes = array(spikes, dtype=int)
        pycuda.driver.memcpy_htod(int(self.gpu_spikes.gpudata), spikes)
        if self.gpuman.force_sync:
            self.memman.copy_to_device(True)
        args = self.gpu_args+(numpy.int64(len(spikes)),)
        self.gpu_func.prepared_call(self.grid, *args)
        if self.gpuman.force_sync:
            self.memman.copy_to_host(True)
