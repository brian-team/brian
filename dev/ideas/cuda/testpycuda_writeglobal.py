import pycuda.autoinit as autoinit
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray
from pycuda import gpuarray
import numpy, pylab, time

N = 1000000
x0 = 3.2
block = (512,1,1)
grid = (int(N/512)+1,1)

mod = drv.SourceModule("""
__global__ void threshold(double *x, double x0, int *J, unsigned int *global_j, int N)
{
 int i = blockIdx.x * blockDim.x + threadIdx.x;
 if(x[i]>x0 && i<N){
  unsigned int j = atomicInc(global_j, N);
  J[j] = i;
 }
}
""")

threshold = mod.get_function("threshold")

v = gpuarray.to_gpu(numpy.random.randn(N))

J = drv.mem_alloc(4*N)
global_j = drv.mem_alloc(4)

Jret = numpy.zeros(N, dtype=int)
jret = numpy.zeros(1, dtype=numpy.uint32)

drv.memcpy_htod(J, numpy.zeros(N, dtype=int))
drv.memcpy_htod(global_j, numpy.zeros(1, dtype=numpy.uint32))

start = time.time()
threshold(v, numpy.float64(x0), J, global_j, numpy.int32(N),
            block=block, grid=grid)
drv.memcpy_dtoh(jret, global_j)
Jret = Jret[:jret[0]]
drv.memcpy_dtoh(Jret, J)
print 'GPU time without sorting:', time.time()-start
#Jret = Jret[:jret[0]]
Jret.sort()
print 'GPU time with sorting on CPU:', time.time()-start

#print Jret#[Jret<N]

start = time.time()
Jcpu = numpy.where(v.get()>x0)[0]
print'CPU time with numpy:', time.time()-start

print all(Jret==Jcpu)
print len(Jret), float(len(Jret))/N