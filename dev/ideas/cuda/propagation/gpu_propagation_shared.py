from brian import *
import pycuda

__all__ = ['atomic_float_code',
           'BaseGPUConnection',
           ]

def atomic_float_code():
    vmaj, vmin = pycuda.autoinit.device.compute_capability()
    code = ''
    if vmaj<2:
        code += '''
        // CUDA manual version
        __device__ float atomicAdd(float* address, float val)
        {
            unsigned int* address_as_ull = (unsigned int*)address;
            unsigned int old = *address_as_u, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_u, assumed,
                        __float_as_int(val + __int_as_float(assumed)));
            } while (assumed != old);
            return __int_as_float(old);
        }
        '''
    if vmaj>1 or vmin>3:
        code += '''
        // CUDA manual version
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
        '''
    return code


class BaseGPUConnection(Connection):
    def __init__(self, *args, **kwds):
        self.usefloat = kwds.pop('use_float', False)
        if self.usefloat:
            self.scalar = 'float'
        else:
            self.scalar = 'double'
        super(BaseGPUConnection, self).__init__(*args, **kwds)
    def gpu_scalar_array(self, arr):
        if self.usefloat:
            return pycuda.gpuarray.to_gpu(array(arr, dtype=float32))
        else:
            return pycuda.gpuarray.to_gpu(arr)
