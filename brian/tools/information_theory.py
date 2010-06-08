'''
Information theory measures.

Adapted by R Brette from several papers (see in the code).
Uses the ANN wrapper in scikits.
'''
import scikits.ann as ann
from scipy.special import gamma, psi
from scipy.linalg import det, inv
from scipy import *

__all__ = ['nearest_distances', 'entropy', 'mutual_information', 'entropy_gaussian',
         'mutual_information2']

def nearest_distances(X, k=1):
    '''
    X = array(N,M)
    N = number of points
    M = number of dimensions
    
    returns the squared distance to the kth nearest neighbor for every point in X
    '''
    ktree = ann.kdtree(X)
    _, d = ktree.knn(X, k + 1) # the first nearest neighbor is itself
    return d[:, -1] # returns the distance to the kth nearest neighbor

def entropy_gaussian(C):
    '''
    Entropy of a gaussian variable with covariance matrix C
    '''
    if isscalar(C): # C is the variance
        return .5 * (1 + log(2 * pi)) + .5 * log(C)
    else:
        n = C.shape[0] # dimension
        return .5 * n * (1 + log(2 * pi)) + .5 * log(abs(det(C)))

def entropy(X, k=1):
    '''
    Returns the entropy of X,
    given as array X = array(n,dx)
    where
      n = number of samples
      dx = number of dimensions
    
    Optionally:
      k = number of nearest neighbors for density estimation
    '''
    # Distance to kth nearest neighbor
    r = nearest_distances(X, k) # squared distances
    n, d = X.shape
    volume_unit_ball = (pi ** (.5 * d)) / gamma(.5 * d + 1)
    '''
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
     
    return .5*d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    '''
    '''
    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    '''
    return .5 * d * mean(log(r)) + log(volume_unit_ball) + psi(n) - psi(k)

def mutual_information(X, Y, k=1):
    '''
    Returns the mutual information between X and Y.
    Each variable is a matrix X = array(n,dx)
    where
      n = number of samples
      dx,dy = number of dimensions
    
    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation
    '''
    return entropy(X) + entropy(Y) - entropy(hstack((X, Y)), k=k)

def mutual_information2(X, p):
    '''
    Mutual information of continuous variable X and discrete
    variable Y taking p values with equal probabilities.
    X is a matrix (n,d) such that X[(n/p)*i:(n/p)*(i+1),:] are the samples
    for the same value of Y.
    
    Maybe to a principal component analysis?
    (problem of highly correlated variables)
    
    Method:
    * Empirical average I=<log(p(X|Y)/p(X))>
    * p(X) and p(X|Y) are estimated with Gaussian assumptions
    '''
    X = X / X.max()
    n, d = X.shape
    M = mean(X, axis=0).reshape((d, 1))
    C = cov(X.T)
    C += C.max()*.001 * eye(d) # avoids degeneracy (hack!)
    invC = inv(C)
    m = int(n / p) # number of samples / value of Y
    I = 0.
    for i in range(p):
        Z = X[m * i:m * (i + 1), :] # same Y
        MZ = mean(Z, axis=0).reshape((d, 1))
        CZ = cov(Z.T)
        CZ += CZ.max()*.001 * eye(d) # avoids degeneracy (hack!)
        #try:
        invCZ = inv(CZ)
        #except LinAlgError: # singular matrix!
        #    
        #A=invC-invCZ
        Ii = 0.
        for j in range(m):
            x = X[m * i + j, :].reshape((d, 1))
            Ii = dot(dot(x.T - M.T, invC), x - M) - dot(dot(x.T - MZ.T, invCZ), x - MZ)
            #Ii+=dot(dot(x.T,A),x)
        Ii = .5 * (log(det(C * invCZ)) + 1) + Ii / m
        I += Ii
    I = I / p
    return I[0, 0]

if __name__ == '__main__':
    '''
    Testing against correlated Gaussian variables
    (analytical results are known)
    '''
    from scipy import *
    # Entropy of a 15-dimensional gaussian variable
    n = 10000
    d = 15
    P = randn(d, d) # change of variables
    C = dot(P, P.T)
    Y = randn(d, n)
    X = dot(P, Y)
    H = entropy_gaussian(C)
    Hest = entropy(X.T, k=5)
    print Hest, H

    # Mutual information between two correlated gaussian variables
    P = randn(2, 2)
    C = dot(P, P.T)
    U = randn(2, n)
    Z = dot(P, U).T
    X = Z[:, 0]
    X = X.reshape(len(X), 1)
    Y = Z[:, 1]
    Y = Y.reshape(len(Y), 1)
    # in bits
    print mutual_information(X, Y, k=5) / log(2.), \
          (entropy_gaussian(C[0, 0]) + entropy_gaussian(C[1, 1]) - entropy_gaussian(C)) / log(2.)
