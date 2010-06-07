'''
Attempting to use chag:pp CUDA library with pycuda or with weave.
This doesn't work.

Message on PyCUDA list follows:

On Dienstag 29 September 2009, Dan Goodman wrote:
> > Hi all,
> > 
> > Has anyone tried using pycuda with a library of GPU code such as CUDPP
> > [1] or chag:pp [2]? How do you use these in pycuda? Is it just a matter
> > of passing some header and lib file directories to nvcc via the
> > options=... keyword for SourceModule?

SourceModule won't work because it's meant only for device-side code, and most 
of these libraries expose host-side interfaces.

If the library is written with the driver API in mind, it's as simple as 
wrapping it into Python using, e.g. Boost.Python, cython, etc. If it is using 
the CUDA "runtime" API, it's not quite as simple, because then Nvidia's 
idiotic driver/runtime dichotomy kicks in and prevents direct communication. 
Luckily, however, I learned at the Nvidia conference last week that a) they're 
aware that this is causing problems, and b) this issue will go away in CUDA 
3.0, which will be released along with their new hardware.

HTH,
Andreas

'''

import pycuda.autoinit as autoinit
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray
from pycuda import gpuarray
import bisect
import numpy, pylab, time, random
from scipy import weave

N=1024

mod=drv.SourceModule('''
#include <chag/pp/compact.cuh>

__global__ void test()
{
 int i = blockIdx.x * blockDim.x + threadIdx.x;
 i = i+1;
}
''', options=[r'-I"C:\Documents and Settings\goodman.CELERI\Bureau\source-20090929"'],
     no_extern_c=True)

#x = numpy.random.randn(N)
#xgpu = gpuarray.to_gpu(x)
#ygpu = gpuarray.to_gpu(numpy.zeros(x.shape))
#vec = gpuarray.to_gpu(numpy.zeros(1, dtype=int))
#z = x[x>=0.0]
#
#xgpu_start = int(xgpu.gpudata)
#xgpu_end = xgpu_start+N*8
#ygpu_start = int(ygpu.gpudata)
#vec_start = int(vec.gpudata)
#
#src = '''
#struct Predicate
#{
#    __device__ bool operator() (double value) const
#    {
#        return value>0.0;
#    }
#}
#
#pp::compact(
#    (double *)xgpu_start,              /* Input start pointer */
#    (double *)xgpu_end,     /* Input end pointer */
#    (double *)ygpu_start,              /* Output start pointer */
#    vec_start,            /* Storage for valid element count */
#    Predicate()             /* Predicate */
#    );      
#'''
#
#weave.inline(src, ['xgpu_start', 'xgpu_end', 'ygpu_start', 'vec_start'],
#             compiler='gcc',#msvc works too
#             headers=[
#                      '<chag/pp/compact.cuh>'],
#             include_dirs=['C:\\CUDA\\include',
#                           r'C:\Documents and Settings\goodman.CELERI\Bureau\source-20090929'],
#             libraries=[],
#             library_dirs=['C:\\CUDA\\lib'],
#             )
