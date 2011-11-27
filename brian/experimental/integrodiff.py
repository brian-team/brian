'''
Conversion from integral form to differential form.
See BEP-5.
 
TODO:
* maximum rank
* better function name
* discrete time version
* rescale X0 to avoid numerical problems
* automatic determination of T?
'''
import re
import inspect
from brian.units import *
from brian.stdunits import *
from brian.inspection import get_identifiers
from brian.utils.autodiff import *
from brian.equations import *
from scipy import linalg

def integral2differential(expr, T=20 * ms, level=0, N=20, suffix=None, matrix_output=False):
    '''
    Example:
      eqs,w=integral2differential('g(t)=t*exp(-t/tau)')
      M,nvar,w=integral2differential('g(t)=t*exp(-t/tau)',matrix_output=True)
      
    Returns an Equations object corresponding to the time-invariant linear system specified
    by the impulse response g(t), and the value w to generate the impulse response:
    g_in->g_in+w.
    
    If matrix_output is True, returns the matrix of the corresponding differential system, the
    index nvar of the variable and the initial condition w=x_nvar(0).
    
    T is the interval over which the function is calculated.
    N is the number of points chosen in that interval.
    level is the frame level where the expression is defined.
    suffix is a string added to internal variable names (default: unique string).
    '''
    # Expression matching
    varname, time, RHS = re.search('\s*(\w+)\s*\(\s*(\w+)\s*\)\s*=\s*(.+)\s*', expr).groups()

    # Build the namespace
    frame = inspect.stack()[level + 1][0]
    global_namespace, local_namespace = frame.f_globals, frame.f_locals
    # Find external objects
    vars = list(get_identifiers(RHS))
    namespace = {}
    for var in vars:
        if var == time: # time variable
            pass
        elif var in local_namespace: #local
            namespace[var] = local_namespace[var]
        elif var in global_namespace: #global
            namespace[var] = global_namespace[var]
        elif var in globals(): # typically units
            namespace[var] = globals()[var]

    # Convert to a function
    f = eval('lambda ' + time + ':' + RHS, namespace)

    # Unit
    unit = get_unit(f(rand()*second)).name

    # Pick N points
    t = rand(N) * T

    # Calculate derivatives and find rank
    n = 0
    rank = 0
    M = f(t).reshape(N, 1)
    while rank == n:
        n += 1
        dfn = differentiate(f, t, order=n).reshape(N, 1)
        x, _, rank, _ = linalg.lstsq(M, dfn)
        if rank == n:
            M = hstack([M, dfn])
            oldx = x
    # oldx expresses dfn as a function of df0,..,dfn-1 (n=rank)

    # Find initial condition
    X0 = array([differentiate(f, 0 * ms, order=n) for n in range(rank)])

    # Rescaling DOES NOT WORK
    #R=ones(rank)
    #for i in range(rank):
    #    if X0[i]!=0.:
    #        R[i]=1./X0[i]
    #    else:
    #        R[i]=1.
    #R=diag(R)
    #X0=dot(R,X0)
    #oldx=dot(R,oldx)

    # Build A
    A = diag(ones(rank - 1), 1)
    A[-1, :] = oldx.reshape(1, rank)

    # Find Q=P^{-1}
    Q = eye(rank)
    if X0[0] == 0.: # continuous g, spikes act on last variable: x->x+1
        Q[:, -1] = X0
        nvar = rank - 1
        w = 1.
        # Exact inversion
        P = eye(rank)
        P[:-1, -1] = -X0[:-1] / X0[-1] # Has to be !=0 !!
        P[-1, -1] = 1. / X0[-1]
    else: # discontinuous g, spikes act on first variable: x->x+g(0)
        Q[:, 0] = X0
        nvar = 0
        w = X0[0]
        P = linalg.inv(Q)

    M = dot(dot(P, A), Q)
    #M=dot(linalg.inv(R),dot(M,R))

    # Turn into string
    # Set variable names
    if rank < 5:
        names = [varname] + ['x', 'y', 'z'][:rank - 1]
    else:
        names = [varname] + ['x' + str(i) for i in range(rank - 1)]

    # Add suffix
    if suffix is None:
        suffix = unique_id()
    names[1:] = [name + suffix for name in names[1:]]

    # Build string
    eqs = []
    for i in range(rank):
        eqs.append('d' + names[i] + '/dt=' + '+'.join([str(x) + '*' + name for x, name in zip(M[i, :], names) if x != 0.]) +
                   ' : ' + str(unit))
    eqs.append(varname + '_in=' + names[nvar]) # alias
    eq_string = '\n'.join(eqs).replace('+-', '-')

    if matrix_output:
        return M, nvar, w
    else:
        return Equations(eq_string), w

if __name__ == '__main__':
    from brian import *
    from scipy import linalg
    tau = 10 * ms
    tau2 = 5 * ms
    freq = 350 * Hz
    # The gammatone example does not seem to work for higher orders
    # probably a numerical problem; use a rescaling matrix for X0?
    f = lambda t:(t / tau) ** 1 * exp(-t / tau) * cos(2 * pi * freq * t)
    A, nvar, w = integral2differential('g(t)=(t/tau)**1*exp(-t/tau)*cos(2*pi*freq*t)', suffix='',
                                   matrix_output=True)
    eq, w = integral2differential('g(t)=(t/tau)**1*exp(-t/tau)*cos(2*pi*freq*t)', suffix='',
                                   matrix_output=False)
    print eq
    #f=lambda t:exp(-t/tau)-exp(-t/tau2)*cos(2*pi*t/tau)
    #A,nvar,w=integral2differential('g(t)=exp(-t/tau)-exp(-t/tau2)*cos(2*pi*t/tau)',
    #                                matrix_output=True)
    print A, nvar, w
    for t in range(10):
        t = t * 1 * ms
        print linalg.expm(A * t)[0, nvar] * w, f(t)
    #t=arange(50)*.5*ms
    #plot(t,f(t))
    #show()
