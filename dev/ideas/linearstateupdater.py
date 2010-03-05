from brian import *
from scipy.linalg import expm

def trapezoidal_integration(f, a, b, n):
    """
    Computes numerically the integral of the function f between a and b
    with a step dt, using the trapezoidal method.
    f(t) must accept vectors of t, and returns a D*n matrix, D is the dimension
    of the output space of f, n is the length of t.
    """
    I = f(a) + f(b)
    dt = (b-a)*1.0/n
    t = dt*arange(1,n)
    I += 2*sum(f(t), axis=0)
    I *= dt/2
    return I

def test_trapezoidal_integration():
    f = lambda x:x**2+3*x+4
    a = 0
    b = 1
    I = trapezoidal_integration(f, a, b, 1000)
    print I, 1./3+3./2+4

def compute_exact(X0, b, t):
    a = array([[X0[0]+b[1], b[0]-X0[1]],[X0[1]-b[0], b[1]+X0[0]]])
    c = array([[cos(t)],[sin(t)]])
    b2 = array([[-b[1]],[b[0]]])
    X = dot(a,c)+b2
    return X.transpose()[0]

def compute_exp(X0, b, t, dt):
    eAt = expm(A*t)
    X = dot(eAt,X0)
    def f(u):
        if isscalar(u):
            return dot(expm(-A*u),b)
        else:
            return array([dot(expm(-A*v),b) for v in u])
    X += dot(eAt, trapezoidal_integration(f, 0, t, dt))
    return X

def compute_euler(X0, b, t, n):
    dt = t/n
    X = zeros((len(X0),n))
    X[:,0] = X0
    for i in range(n-1):
        X[:,i+1] = X[:,i] + dt*(dot(A,X[:,i])+b)
    return X

def compute_exactstep(X0, b, t, n, nint):
    dt = t*1.0/(n-1)
    X = zeros((len(X0),n))
    X[:,0] = X0
    
    eAt = expm(A*dt)
    def f(u):
        if isscalar(u):
            return dot(expm(-A*u),b)
        else:
            return array([dot(expm(-A*v),b) for v in u])
    eAtint = dot(eAt, trapezoidal_integration(f, 0, dt, nint))
    
    for i in range(n-1):
        X[:,i+1] = dot(eAt, X[:,i]) + eAtint
    return X

if __name__ == '__main__':
    """
    Simulation of the system X' = AX+b, X(0) = X0, between 0 and T, with
    various integration schemes.
    """
    
    X0 = randn(2)
    b = randn(2)
    A = array([[0,-1],[1,0]])
    T = 10.0
    dt = .0001 # stepsize of the network simulation
    n = int(T/dt)+1
    
#    X_euler = compute_euler(X0, b, T, dt)
#    print "euler", X_euler[:,-1]
#    
#    X_exp = compute_exp(X0, b, T, dt)
#    print "expm ", X_exp

    X_exact = zeros((2,n))
    for i in xrange(0,n):
        X_exact[:,i] = compute_exact(X0, b, i*dt)
    print "exact", X_exact[:,-1]
    
    for nint in [1,10,100,1000]:
        X_exactstep = compute_exactstep(X0, b, T, n, nint)
        print "exexp", X_exactstep[:,-1]
        
        error = sum((X_exact - X_exactstep)**2, axis=0)
        print nint, "error:", max(error)
#        print error.shape
#        plot(error)
#    show()
    