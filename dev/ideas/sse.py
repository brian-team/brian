'''
Testing whether or not GCC generates code using SSE extensions, and if not,
can we make it so that it does.
'''
from brian import *
from scipy import weave
import time

dtype = float32
repeats = 100
N = 1000000
complexity = 10
x = zeros(N, dtype=dtype)
y = ones(N, dtype=dtype)

if dtype is float32:
    supp = '''
    typedef float scalar;
    typedef float vec __attribute__ ((vector_size (16)));
    '''
else:
    supp = '''
    typedef double scalar;
    typedef double vec __attribute__ ((vector_size (16)));
    '''

code = '''
for(int j=0;j<repeats;j++)
  for(int i=0;i<N;i++)
  {
    scalar &vx = x[i];
    scalar &vy = y[i];
    vx += VY;
  }
'''.replace('VY','*'.join(['vy']*complexity))

if dtype is float32:
    scalarsize = 4
    vecsize = 4
else:
    scalarsize = 8
    vecsize = 2
code2 = '''
int i0, i1;
i0 = ((unsigned int)x%16)/SCALARSIZE;
i1 = ((N-i0)/VECSIZE)*VECSIZE;
for(int j=0;j<repeats;j++)
{
  for(int i=0;i<i0;i++)
  {
    scalar &vx = x[i];
    scalar &vy = y[i];
    vx += VY;
  }
  for(int i=i0;i<i1;i+=VECSIZE)
  {
    vec &vx = *(vec*)(x+i);
    vec &vy = *(vec*)(y+i);
    vx += VY;
  }
  for(int i=i1;i<N;i++)
  {
    scalar &vx = x[i];
    scalar &vy = y[i];
    vx += VY;
  }
}
'''.replace('VY','*'.join(['vy']*complexity)).replace('VECSIZE', str(vecsize)).replace('SCALARSIZE', str(scalarsize))

def f(N, repeats, force):
    start = time.time()
    weave.inline(code, ['x', 'y', 'N', 'repeats'], compiler='gcc',
                 extra_compile_args=[
                    '-march=native',
                    '-O3',
                    ],
                 support_code=supp,
                 verbose=2*force,
                 force=force,
                 )
    end = time.time()
    return end-start

f(0, 0, True)
print 'Time taken:', f(N, repeats, False)
print 'Correct:', (x==repeats).all()