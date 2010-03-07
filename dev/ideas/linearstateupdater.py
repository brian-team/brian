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
    t = a+dt*arange(1,n)
    I += 2*sum(f(t), axis=0)
    I *= dt/2
    return I

def simpson_integration(f, a, b, n):
    """
    Computes numerically the integral of the function f between a and b
    with a step dt, using the Simpson method.
    f(t) must accept vectors of t, and returns a D*n matrix, D is the dimension
    of the output space of f, n is the length of t.
    """
    n = 2*int(n/2)
    I = f(a) + f(b)
    dt = (b-a)*1.0/n
    te = a+dt*arange(2,n,2)
    to = a+dt*arange(1,n,2)
    I += 2*sum(f(te), axis=0)
    I += 4*sum(f(to), axis=0)
    I *= dt/3
    return I

def test_integration():
    a = randn(1)
    b = randn(1)
    c = randn(1)
    A = randn(1)
    B = A+randn(1)**2
    f = lambda x:a*exp(x)+b*x+c
    Itrap = trapezoidal_integration(f, A, B, 100)
    Isimp = simpson_integration(f, A, B, 100)
    Itrue = a*(exp(B)-exp(A))+b*(B**2/2-A**2/2)+c*(B-A)
    print abs(Itrap-Itrue), abs(Isimp-Itrue)

def compute_exact(X0, b, t):
#    a = array([[X0[0]+b[1], b[0]-X0[1]],[X0[1]-b[0], b[1]+X0[0]]])
#    c = array([[cos(t)],[sin(t)]])
#    b2 = array([[-b[1]],[b[0]]])
#    X = dot(a,c)+b2
#    return X.transpose()[0]
    X = array([(X0+b)*exp(t)-b])
    return X

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

def test():
    X0 = randn(1)
    b = randn(1)
    T = 10.0
    dt = .0001 # stepsize of the network simulation
    n = int(T/dt)+1
    nint = 100
    
    def f(u):
        if isscalar(u):
            return exp(-u)*b
        return reshape(exp(-u)*b, (len(u),1))
    I = simpson_integration(f, 0, dt, nint)
    print abs(I-b*(1-exp(-dt)))

if __name__ == '__main__':
    """
    Simulation of the system X' = AX+b, X(0) = X0, between 0 and T, with
    different integration schemes.
    """
    
#    A = array([[0,-1],[1,0]])
    
    A = array([[1]])
    
    D = len(A)
    X0 = randn(D)
    b = randn(D)
    T = 10.0
    dt = .0001 # stepsize of the network simulation
    n = int(T/dt)+1
    
#    X_euler = compute_euler(X0, b, T, dt)
#    print "euler", X_euler[:,-1]
#    
#    X_exp = compute_exp(X0, b, T, dt)
#    print "expm ", X_exp

    X_exact = zeros((D,n))
    for i in xrange(0,n):
        X_exact[:,i] = compute_exact(X0, b, i*dt)
    print "exact", X_exact[:,-1]
    
    nints = linspace(1, 1001, 20)
    errors = zeros(len(nints))
    for i in xrange(len(nints)):
        nint = int(nints[i])
        X_exactstep = compute_exactstep(X0, b, T, n, nint)
#        print "exexp", X_exactstep[:,-1]

        error = sqrt(sum((X_exact - X_exactstep)**2, axis=0))
        print nint, "error:", max(error)
        
        errors[i] = max(error)
        
    plot(nints, errors)
    show()
    