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
    dt = (b - a) * 1.0 / n
    t = a + dt * arange(1, n)
    I += 2 * sum(f(t), axis=0)
    I *= dt / 2
    return I

def simpson_integration(f, a, b, n):
    """
    Computes numerically the integral of the function f between a and b
    with a step dt, using the Simpson method.
    f(t) must accept vectors of t, and returns a D*n matrix, D is the dimension
    of the output space of f, n is the length of t.
    """
    n = 2 * int(n / 2)
    I = f(a) + f(b)
    dt = (b - a) * 1.0 / n
    te = a + dt * arange(2, n, 2)
    to = a + dt * arange(1, n, 2)
    I += 2 * sum(f(te), axis=0)
    I += 4 * sum(f(to), axis=0)
    I *= dt / 3
    return I

def test_integration():
    a = randn(1)
    b = randn(1)
    c = randn(1)
    A = randn(1)
    B = A + randn(1) ** 2
    f = lambda x:a * exp(x) + b * x + c
    Itrap = trapezoidal_integration(f, A, B, 100)
    Isimp = simpson_integration(f, A, B, 100)
    Itrue = a * (exp(B) - exp(A)) + b * (B ** 2 / 2 - A ** 2 / 2) + c * (B - A)
    print abs(Itrap - Itrue), abs(Isimp - Itrue)

def compute_exact1(X0, b, t):
    a = array([[X0[0] + b[1], b[0] - X0[1]], [X0[1] - b[0], b[1] + X0[0]]])
    c = array([[cos(t)], [sin(t)]])
    b2 = array([[-b[1]], [b[0]]])
    X = dot(a, c) + b2
    return X.transpose()[0]

def compute_exact2(X0, b, t):
    X = array([(X0 + b) * exp(t) - b])
    return X

def compute_euler(X0, A, b, t, n):
    dt = t / n
    X = zeros((len(X0), n + 1))
    X[:, 0] = X0
    for i in range(n):
        X[:, i + 1] = X[:, i] + dt * (dot(A, X[:, i]) + b)
    return X

def compute_Ab(A, b, dt, nint):
    eAt = expm(A * dt)
    def f(u):
        if isscalar(u):
            return dot(expm(-A * u), b)
        else:
            return array([dot(expm(-A * v), b) for v in u])
    eAtint = dot(eAt, trapezoidal_integration(f, 0, dt, nint))
    return eAt, eAtint

def compute_exactstep(X0, A, b, t, n, nint):
    dt = t / n
    X = zeros((len(X0), n + 1))
    X[:, 0] = X0

#    eAt = expm(A*dt)
#    def f(u):
#        if isscalar(u):
#            return dot(expm(-A*u),b)
#        else:
#            return array([dot(expm(-A*v),b) for v in u])
#    eAtint = dot(eAt, trapezoidal_integration(f, 0, dt, nint))

    eAt, eAtint = compute_Ab(A, b, dt, nint)

    for i in range(n):
        X[:, i + 1] = dot(eAt, X[:, i]) + eAtint
    return X

def test_integral():
    a = 1000#randn(1)
    b = 1000#randn(1)
    dt = .0001
    nint = 10

    def f(u):
        if isscalar(u):
            return exp(-a * u) * b
        return reshape(exp(-a * u) * b, (len(u), 1))

    I = simpson_integration(f, 0, dt, nint)
    print "actual integration error       ", abs(I[0] - b / a * (1 - exp(-a * dt)))
    print "upper bound integration error  ", dt ** 5 / (180 * nint ** (4)) * exp(abs(a) * dt) * abs(b) * a ** 4

def test_sim():
    """
    Simulation of the system X' = AX+b, X(0) = X0, between 0 and T, with
    different integration schemes.
    """
    A = array([[0, -1], [1, 0]])
    D = len(A)
    b = 10 * randn(D)
    X0 = 10 * randn(D)

    T = 1.0
    dt = .0001
    n = 10000
    nint = 100

    X_exact = zeros((D, n + 1))
    for i in xrange(0, n + 1):
        X_exact[:, i] = compute_exact1(X0, b, i * dt)
    print "exact", X_exact[:, -1]

#    X_euler = compute_euler(X0, A, b, T, n)
#    print "euler error:      ", max(abs(X_euler[:,-1]-X_exact[:,-1]))

    X_exactstep = compute_exactstep(X0, A, b, T, n, nint)
    print "exact steps error:", max(abs(X_exactstep[:, -1] - X_exact[:, -1]))

def precision(dt, n, D, a, b):
    return dt ** 5 / (180 * n ** 4) * sqrt(D) * a ** 4 * exp(2 * a * dt) * b

def test_precision():
    dt = 1e-4
    n = 100
    D = 10
    a = b = 1000
    print precision(dt, n, D, a, b)

def is_equal(x, y, epsilon):
    return (abs(x - y) < abs(x) * epsilon).all()

if __name__ == '__main__':
#    test_sim()

    epsilon = 1.
    while 1. + epsilon > 1.:
        epsilon /= 2
    epsilon *= 2.

    A = array([[0, -1], [1, 0]])
    D = len(A)
    b = 10 * randn(D)
    dt = .0001

    nint = 16
    btilde2 = compute_Ab(A, b, dt, nint)[1]
    for i in xrange(12):
        nint *= 2
        btilde = compute_Ab(A, b, dt, nint)[1]
        print nint, max(abs(btilde - btilde2)), max(abs(btilde)) * epsilon
        btilde2 = btilde
