from brian import *
from scipy import weave
import time

N = 1000

eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(10*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

P = NeuronGroup(N, eqs)

Scopy = P._S.copy()

run(20*ms)

print P._S[:, 0]

B = P._state_updater.B
A = P._state_updater.A
C = P._state_updater._C

S = Scopy.copy()
T = S.copy()

def withdot(nsteps):
    for _ in xrange(nsteps):
        S[:] = dot(A, S)
    #    dot(A, S, out=S) # doesn't seem to work
    #    # but this does work (no quicker though)
    #    dot(A, S, out=T)
    #    S[:] = T
        add(S, C, S)
        
withdot(200)
print S[:, 0]

print S.flatten()

print A
print C

print A.flatten()

code = '''
double *v = S;
double *ge = S+N;
double *gi = S+2*N;
for(int i=0; i<N; ++i)
{
    double v_next = A[0]*v[i]+A[1]*ge[i]+A[2]*gi[i]+C[0];
    double ge_next = A[3]*v[i]+A[4]*ge[i]+A[5]*gi[i]+C[1];
    double gi_next = A[6]*v[i]+A[7]*ge[i]+A[8]*gi[i]+C[2];
    v[i] = v_next;
    ge[i] = ge_next;
    gi[i] = gi_next;
}
'''

def withweave(nsteps):
    for _ in xrange(nsteps):
        weave.inline(code, ['S', 'N', 'A', 'C'],
                     compiler='gcc',
                     extra_compile_args=['-O3', '-ffast-math',
                                         '-march=native',
                                         ])

S = Scopy.copy()
withweave(200)
print S[:, 0]

# verified that it works, now do speed test

numsteps = 100000000/N

print
print 'N:', N
print 'numsteps:', numsteps

start = time.time()
withdot(numsteps)
print 'With dot:', time.time()-start

start = time.time()
withweave(numsteps)
print 'With weave:', time.time()-start
