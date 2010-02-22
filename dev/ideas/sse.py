'''
Testing whether or not GCC generates code using SSE extensions, and if not,
can we make it so that it does.
'''
from brian import *
from scipy import weave
import time

repeats = 100
N = 1000000
x = zeros(N)
y = ones(N)

supp = '''
typedef double v2df __attribute__ ((vector_size (16)));
//typedef union { double s[2]; v2df v; } v2df_u;
'''

code = '''
for(int j=0;j<repeats;j++)
  for(int i=0;i<N;i++)
  {
    double &vx = x[i];
    double &vy = y[i];
    vx += VY;
  }
'''.replace('VY','*'.join(['vy']*100))

code2 = '''
int i0, i1;
i0 = ((unsigned int)x%16)/8;
i1 = ((N-i0)/2)*2;
for(int j=0;j<repeats;j++)
{
  for(int i=0;i<i0;i++)
  {
    double &vx = x[i];
    double &vy = y[i];
    vx += VY;
  }
  for(int i=i0;i<i1;i+=2)
  {
    v2df &vx = *(v2df*)(x+i);
    v2df &vy = *(v2df*)(y+i);
    vx += VY;
  }
  for(int i=i1;i<N;i++)
  {
    double &vx = x[i];
    double &vy = y[i];
    vx += VY;
  }
}
'''.replace('VY','*'.join(['vy']*100))

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