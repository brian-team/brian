'''
'''
# TODO: support for GPUBufferedArray?
from numpy import *
import pycuda
#import pycuda.autoinit as autoinit
import pycuda.driver as drv
import pycuda.compiler
from pycuda import gpuarray
from brian.experimental.cuda.buffering import *
import re
from filterbank import Filterbank, RestructureFilterbank
from gputools import *
import gc

__all__ = ['LinearFilterbank']

class LinearFilterbank(Filterbank):
    '''
    Generalised parallel linear filterbank
    
    This filterbank allows you to construct a chain of linear filters in
    a bank so that each channel in the bank has its own filters. You pass
    the (b,a) parameters for the filter in the format specified in the
    function ``apply_linear_filterbank``.
    
    Note that there are additional GPU specific options:
    
    ``precision``
        Should be single for older GPUs.
    ``forcesync``
        Should be set to true to force a copy of GPU->CPU after each
        filter step, but not if the output is being used in ongoing
        GPU computations. By default this is ``True`` for compatibility.
    ``pagelocked_mem``
        Allocate faster pagelocked memory for GPU->CPU copies (doubles
        copy rate).
    ``unroll_filterorder``
        Whether or not to unroll the loop for the filter order, normally
        this should be done for small filter orders. By default, it is done
        if the filter order is less than or equal to 32.
    '''
    def __init__(self, source, b, a, samplerate=None,
                 precision='double', forcesync=True, pagelocked_mem=True, unroll_filterorder=None):
        # Automatically duplicate mono input to fit the desired output shape
        if b.shape[0]!=source.nchannels:
            if source.nchannels!=1:
                raise ValueError('Can only automatically duplicate source channels for mono sources, use RestructureFilterbank.')
            source = RestructureFilterbank(source, b.shape[0])
        Filterbank.__init__(self, source)
        if pycuda.context is None:
            set_gpu_device(0)
        self.precision=precision
        if self.precision=='double':
            self.precision_dtype=float64
        else:
            self.precision_dtype=float32
        self.forcesync=forcesync
        self.pagelocked_mem=pagelocked_mem
        n, m, p=b.shape
        self.filt_b=b
        self.filt_a=a
        filt_b_gpu=array(b, dtype=self.precision_dtype)
        filt_a_gpu=array(a, dtype=self.precision_dtype)
        filt_state=zeros((n, m-1, p), dtype=self.precision_dtype)
        if pagelocked_mem:
            filt_y=drv.pagelocked_zeros((n,), dtype=self.precision_dtype)
            self.pre_x=drv.pagelocked_zeros((n,), dtype=self.precision_dtype)
        else:
            filt_y=zeros(n, dtype=self.precision_dtype)
            self.pre_x=zeros(n, dtype=self.precision_dtype)
        self.filt_b_gpu=gpuarray.to_gpu(filt_b_gpu.T.flatten()) # transform to Fortran order for better GPU mem
        self.filt_a_gpu=gpuarray.to_gpu(filt_a_gpu.T.flatten()) # access speeds
        self.filt_state=gpuarray.to_gpu(filt_state.T.flatten())
        if unroll_filterorder is None:
            if m<=32:
                unroll_filterorder = True
            else:
                unroll_filterorder = False
        # TODO: improve code, check memory access patterns, maybe use local memory
        code='''
        #define x(s,i) _x[(s)*n+(i)]
        #define y(s,i) _y[(s)*n+(i)]
        #define a(i,j,k) _a[(i)+(j)*n+(k)*n*m]
        #define b(i,j,k) _b[(i)+(j)*n+(k)*n*m]
        #define zi(i,j,k) _zi[(i)+(j)*n+(k)*n*(m-1)]
        __global__ void filt(SCALAR *_b, SCALAR *_a, SCALAR *_x, SCALAR *_zi, SCALAR *_y, int numsamples)
        {
            int j = blockIdx.x * blockDim.x + threadIdx.x;
            if(j>=n) return;
            for(int s=0; s<numsamples; s++)
            {
        '''
        for k in range(p):
            loopcode='''
            y(s,j) = b(j,0,k)*x(s,j) + zi(j,0,k);
            '''
            if unroll_filterorder:
                for i in range(m-2):
                    loopcode+=re.sub('\\bi\\b', str(i), '''
                    zi(j,i,k) = b(j,i+1,k)*x(s,j) + zi(j,i+1,k) - a(j,i+1,k)*y(s,j);
                    ''')
            else:
                loopcode+='''
                for(int i=0;i<m-2;i++)
                    zi(j,i,k) = b(j,i+1,k)*x(s,j) + zi(j,i+1,k) - a(j,i+1,k)*y(s,j);
                '''
            loopcode+='''
            zi(j,m-2,k) = b(j,m-1,k)*x(s,j) - a(j,m-1,k)*y(s,j);
            '''
            if k<p-1:
                loopcode+='''
                x(s,j) = y(s,j);
                '''
            loopcode=re.sub('\\bk\\b', str(k), loopcode)
            code+=loopcode
        code+='''
            }
        }
        '''
        code=code.replace('SCALAR', self.precision)
        code=re.sub("\\bp\\b", str(p), code) #replace the variable by their values
        code=re.sub("\\bm\\b", str(m), code)
        code=re.sub("\\bn\\b", str(n), code)
        #print code
        self.gpu_mod=pycuda.compiler.SourceModule(code)
        self.gpu_filt_func=self.gpu_mod.get_function("filt")
        blocksize=256
        if n<blocksize:
            blocksize=n
        if n%blocksize==0:
            gridsize=n/blocksize
        else:
            gridsize=n/blocksize+1
        self.block=(blocksize, 1, 1)
        self.grid=(gridsize, 1)
        self.gpu_filt_func.prepare((intp, intp, intp, intp, intp, int32), self.block)
        self._has_run_once=False

    def reset(self):
        self.buffer_init()

    def buffer_init(self):
        Filterbank.buffer_init(self)
        self.filt_state.set(zeros(self.filt_state.shape, dtype=self.filt_state.dtype))
    
    def buffer_apply(self, input):
        # TODO: buffer apply to a large input may cause a launch timeout, need to buffer in
        # smaller chunks if this is the case
        b = self.filt_b_gpu
        a = self.filt_a_gpu
        zi = self.filt_state
        if not hasattr(self, 'filt_x_gpu') or input.size!=self.filt_x_gpu.size:
            self._desiredshape = input.shape
            self._has_run_once = False
            self.filt_x_gpu = gpuarray.to_gpu(input.flatten())
            self.filt_y_gpu = gpuarray.empty_like(self.filt_x_gpu)
        else:
            self.filt_x_gpu.set(input.flatten())
        filt_x_gpu = self.filt_x_gpu
        filt_y_gpu = self.filt_y_gpu
        if self._has_run_once:
            self.gpu_filt_func.launch_grid(*self.grid)
        else:
            self.gpu_filt_func.prepared_call(self.grid, intp(b.gpudata),
                    intp(a.gpudata), intp(filt_x_gpu.gpudata),
                    intp(zi.gpudata), intp(filt_y_gpu.gpudata), int32(input.shape[0]))
            self._has_run_once = True
        return reshape(filt_y_gpu.get(pagelocked=self.pagelocked_mem), self._desiredshape)
