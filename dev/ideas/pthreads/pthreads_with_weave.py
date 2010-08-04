from brian import *
from scipy import weave
import time
import os

#base, name = os.path.split(__file__)
#print os.path.join(base, 'win32/pthread')
#exit()

N = 10000
x = zeros(N)
nthreads = 1

#if defined(_WIN32) && !defined(__MINGW32__)
  #include "win32/pthread.h"
#else
  #include <pthread.h>
#endif

supp_code = '''
class Work
{
public:
  double *x;
  int istart, iend, N;
  Work(double *x, int istart, int iend, int N) : x(x), istart(istart), iend(iend), N(N) {};
};

void do_work(void* workobj)
{
 Work *w = (Work*)workobj;
 double *x = w->x;
 int istart = w->istart;
 int iend = w->iend;
 int N = w->N;
 for(int i=istart; i<iend; i++)
 {
  x[i] = sin(3.14159*(double)i/(double)N);
 }
}
'''

code = '''
Py_BEGIN_ALLOW_THREADS
Work w = Work(x, 0, N, N);
pthread_t thread;
void *status;
int rc = pthread_create(&thread, NULL, do_work, (void *)&w);
rc = pthread_join(thread, &status);
pthread_exit(NULL);
//do_work((void *)&w);
Py_END_ALLOW_THREADS
'''

start = time.time()
weave.inline(code,
             ['x', 'N', 'nthreads'],
             compiler='mingw32',
             support_code=supp_code,
             headers=['<pthread.h>'],
             extra_compile_args=['-pthread'],
             )
end = time.time()

print end-start

plot(x)
show()
