# TODO: update all of this with the new interface/buffering mechanism

import pycuda
#import pycuda.autoinit as autoinit
import pycuda.driver as drv
import pycuda.compiler
from pycuda import gpuarray
from brian.experimental.cuda.buffering import *
import re
from filterbank import Filterbank
from gputools import *

__all__ = ['LinearFilterbank']

class LinearFilterbank(Filterbank):
    '''
    Generalised parallel linear filterbank
    
    This filterbank allows you to construct a chain of linear filters in
    a bank so that each channel in the bank has its own filters. You pass
    the (b,a) parameters for the filter in the format specified in the
    function ``parallel_lfilter_step``.
    
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
        this should be done but not for very large filter orders.
    '''
    def __init__(self, b, a, samplerate=None,
                 precision='double', forcesync=True, pagelocked_mem=True, unroll_filterorder=True):
        if not use_gpu:
            self.__class__=nongpu_LinearFilterbank
            self.__init__(b, a, samplerate=samplerate)
            return
        if pycuda.context is None:
            set_gpu_device(0)
        self.precision=precision
        if self.precision=='double':
            self.precision_dtype=float64
        else:
            self.precision_dtype=float32
        self.forcesync=forcesync
        self.pagelocked_mem=pagelocked_mem
        self.fs=samplerate
        self.N=b.shape[0]
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
        filt_x=zeros(n, dtype=self.precision_dtype)
        self.filt_b_gpu=gpuarray.to_gpu(filt_b_gpu.T.flatten()) # transform to Fortran order for better GPU mem
        self.filt_a_gpu=gpuarray.to_gpu(filt_a_gpu.T.flatten()) # access speeds
        self.filt_state=gpuarray.to_gpu(filt_state.T.flatten())
        self.filt_x=gpuarray.to_gpu(filt_x)
        self.filt_y=GPUBufferedArray(filt_y)
        #self.filt_y=gpuarray.to_gpu(filt_y)
        code='''
        #define x(i) _x[i]
        #define y(i) _y[i]
        #define a(i,j,k) _a[(i)+(j)*n+(k)*n*m]
        #define b(i,j,k) _b[(i)+(j)*n+(k)*n*m]
        #define zi(i,j,k) _zi[(i)+(j)*n+(k)*n*(m-1)]
        __global__ void filt(SCALAR *_b, SCALAR *_a, SCALAR *_x, SCALAR *_zi, SCALAR *_y)
        {
            int j = blockIdx.x * blockDim.x + threadIdx.x;
            if(j>=n) return;
        '''
        for k in range(p):
            loopcode='''
            y(j) = b(j,0,k)*x(j) + zi(j,0,k);
            '''
            if unroll_filterorder:
                for i in range(m-2):
                    loopcode+=re.sub('\\bi\\b', str(i), '''
                    zi(j,i,k) = b(j,i+1,k)*x(j) + zi(j,i+1,k) - a(j,i+1,k)*y(j);
                    ''')
            else:
                loopcode+='''
                for(int i=0;i<m-2;i++)
                    zi(j,i,k) = b(j,i+1,k)*x(j) + zi(j,i+1,k) - a(j,i+1,k)*y(j);
                '''
            loopcode+='''
            zi(j,m-2,k) = b(j,m-1,k)*x(j) - a(j,m-1,k)*y(j);
            '''
            if k<p-1:
                loopcode+='''
                x(j) = y(j);
                '''
            loopcode=re.sub('\\bk\\b', str(k), loopcode)
            code+=loopcode
        code+='''
        }
        '''
        code=code.replace('SCALAR', self.precision)
        code=re.sub("\\bp\\b", str(p), code) #replace the variable by their values
        code=re.sub("\\bm\\b", str(m), code)
        code=re.sub("\\bn\\b", str(n), code)
        #print code
        self.gpu_mod=pycuda.compiler.SourceModule(code)
        self.gpu_filt_func=self.gpu_mod.get_function("filt")
        blocksize=512#self.maxblocksize
        if n<blocksize:
            blocksize=n
        if n%blocksize==0:
            gridsize=n/blocksize
        else:
            gridsize=n/blocksize+1
        self.block=(blocksize, 1, 1)
        self.grid=(gridsize, 1)
        self.gpu_filt_func.prepare((intp, intp, intp, intp, intp), self.block)
        self._has_run_once=False

    def reset(self):
        self.filt_state.set(zeros(self.filt_state.shape, dtype=self.filt_state.dtype))

    def __len__(self):
        return self.N

    samplerate=property(fget=lambda self:self.fs)

    def timestep(self, input):
        b=self.filt_b_gpu
        a=self.filt_a_gpu
        x=input
        zi=self.filt_state
        y=self.filt_y
        fx=self.filt_x
        if isinstance(x, GPUBufferedArray):
            if not len(x.shape) or x.shape==(1,):
                x.sync_to_cpu()
                newx=empty(self.N, dtype=b.dtype)
                newx[:]=x
                fx.set(newx)
            else:
                drv.memcpy_dtod(fx.gpudata, x.gpu_dev_alloc, fx.size*self.precision_dtype().nbytes)
        else:
            if not isinstance(x, ndarray) or not len(x.shape) or x.shape==(1,):
                # Current version of pycuda doesn't allow .fill(val) method on float64 gpuarrays
                # because it assumed float32 only, so we have to do our own copying here
                px=self.pre_x
                px[:]=x
                x=px
            fx.set(x)
        if self._has_run_once:
            self.gpu_filt_func.launch_grid(*self.grid)
        else:
            self.gpu_filt_func.prepared_call(self.grid, intp(b.gpudata), intp(a.gpudata), intp(fx.gpudata),
                                             intp(zi.gpudata), y.gpu_pointer)
            self._has_run_once=True
        y.changed_gpu_data()
        if self.forcesync:
            y.sync_to_cpu()#might need to turn this on although it slows everything down
        return y

    def apply(self, input):
        return array([self.timestep(x).copy() for x in input])
