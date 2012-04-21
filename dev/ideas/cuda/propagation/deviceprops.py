from pylab import *
import pycuda
import pycuda.autoinit
import pycuda.compiler

dev = pycuda.autoinit.device
devattr = pycuda.autoinit.device.get_attributes()

for k, v in devattr.items():
    print k, v, k.__class__

max_threads_per_block = dev.max_threads_per_block
max_shared_memory_per_block = dev.max_shared_memory_per_block
max_registers_per_block = dev.max_registers_per_block

code = '''
__global__ void func(SCALAR *X)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ SCALAR sh[BLOCKDIM];
    SCALAR x = X[i];
    sh[threadIdx.x] = x;
    LOOP
    x = sh[threadIdx.x];
    X[i] = x;
}
'''

nloop = 10
loopline = 'x = x*x+0.1;'.replace('x', 'sh[threadIdx.x]')
code = code.replace('LOOP', loopline*nloop);

blocksize = max_threads_per_block
code = code.replace('BLOCKDIM', str(blocksize))

code = code.replace('SCALAR', 'double')
#code = code.replace('SCALAR', 'float')

print
print code
print 
print 'max_threads_per_block', max_threads_per_block
print 'max_shared_memory_per_block', max_shared_memory_per_block
print 'max_registers_per_block', max_registers_per_block

module = pycuda.compiler.SourceModule(code)
f = module.get_function('func')
print
print 'func local', f.local_size_bytes
print 'func shared', f.shared_size_bytes
print 'func regs', f.num_regs
print
print 'func local per block', f.local_size_bytes*blocksize
print 'func shared per block', f.shared_size_bytes#*blocksize
print 'func regs per block', f.num_regs*blocksize

#x = linspace(0, 2, 100000, dtype=float64)
