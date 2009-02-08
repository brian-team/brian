'''
Information theory measures.

Adapted by R Brette from several papers (see in the code).
Uses the ANN wrapper in scikits.
'''
import scikits.ann as ann
from scipy.special import gamma,psi
from scipy.linalg import det
from scipy import pi

__all__=['nearest_distances','entropy','mutual_information','entropy_gaussian']

def nearest_distances(X,k=1):
    '''
    X = array(N,M)
    N = number of points
    M = number of dimensions
    
    returns the squared distance to the kth nearest neighbor for every point in X
    '''
    ktree=ann.kdtree(X)
    _,d=ktree.knn(X,k+1) # the first nearest neighbor is itself
    return d[:,-1] # returns the distance to the kth nearest neighbor

def entropy_gaussian(C):
    '''
    Entropy of a gaussian variable with covariance matrix C
    '''
    if isscalar(C): # C is the variance
        return .5*(1+log(2*pi))+.5*log(C)
    else:
        n=C.shape[0] # dimension
        return .5*n*(1+log(2*pi))+.5*log(abs(det(C)))

def entropy(*args,**kwd):
    '''
    Returns the entropy of the variables.
    Each variable is given as an array X = array(n,dx)
    where
      n = number of samples
      dx = number of dimensions
    
    Optionally:
      k = number of nearest neighbors for density estimation
      
    Example: entropy(X), entropy(X,Y), entropy(X,k=5)
    '''
    if len(args)==0:
        raise AttributeError,"No variable!"
    k=kwd.get('k',1)
    X=hstack(args)

    # Distance to kth nearest neighbor
    r=nearest_distances(X,k) # squared distances
    n,d=X.shape
    volume_unit_ball=(pi**(.5*d))/gamma(.5*d+1)
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
    return .5*d*mean(log(r))+log(volume_unit_ball)+psi(n)-psi(k)

def mutual_information(*args,**kwd):
    '''
    Returns the mutual information between any number of variables.
    Each variable is a matrix X = array(n,dx)
    where
      n = number of samples
      dx,dy = number of dimensions
    
    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation

    Example: mutual_information(X,Y), mutual_information(X,Y,Z,k=5)
    '''
    if len(args)<2:
        raise AttributeError,"Mutual information must involve at least 2 variables"
    k=kwd.get('k',1)
    return sum([entropy(X,k=k) for X in args])-entropy(k=k,*args)

if __name__=='__main__':
    '''
    Testing against correlated Gaussian variables
    (analytical results are known)
    '''
    from scipy import *
    # Entropy of a 15-dimensional gaussian variable
    n=10000
    d=15
    P=randn(d,d) # change of variables
    C=dot(P,P.T)
    Y=randn(d,n)
    X=dot(P,Y)
    H=entropy_gaussian(C)
    Hest=entropy(X.T,k=5)
    print Hest,H
    
    # Mutual information between two correlated gaussian variables
    P=randn(2,2)
    C=dot(P,P.T)
    U=randn(2,n)
    Z=dot(P,U).T
    X=Z[:,0]
    X=X.reshape(len(X),1)
    Y=Z[:,1]
    Y=Y.reshape(len(Y),1)
    # in bits
    print mutual_information(X,Y,k=5)/log(2.),\
          (entropy_gaussian(C[0,0])+entropy_gaussian(C[1,1])-entropy_gaussian(C))/log(2.)
    