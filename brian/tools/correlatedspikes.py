# ----------------------------------------------------------------------------------
# Copyright ENS, INRIA, CNRS
# Contributors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
# 
# Brian is a computer program whose purpose is to simulate models
# of biological neural networks.
# 
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# ----------------------------------------------------------------------------------
# 
'''
Generation of correlated spike trains.
Based on the article: Brette, R. (2009). Generation of correlated spike trains.
http://audition.ens.fr/brette/papers/Brette2008NC.html

The models for correlated spike trains are from the paper
but the implemented algorithms are simple ones.

See brian.utils.statistics for correlograms.
'''
from ..threshold import PoissonThreshold, HomogeneousPoissonThreshold
from ..neurongroup import NeuronGroup
from ..equations import Equations
from scipy.special import erf
from scipy.optimize import newton, fmin_tnc
from scipy import *
from ..units import check_units, hertz, second
from ..utils.circular import SpikeContainer
from numpy.random import poisson, binomial, rand, exponential
from random import sample

__all__ = ['rectified_gaussian', 'inv_rectified_gaussian', 'HomogeneousCorrelatedSpikeTrains', \
         'MixtureHomogeneousCorrelatedSpikeTrains',
         'CorrelatedSpikeTrains', 'mixture_process', 'find_mixture']

"""
Utility functions
"""
def rectified_gaussian(mu, sigma):
    '''
    Calculates the mean and standard deviation for a rectified Gaussian distribution.
    mu, sigma: parameters of the original distribution
    Returns mur,sigmar: parameters of the rectified distribution
    '''
    a = 1. + erf(mu / (sigma * (2 ** .5)));

    mur = (sigma / (2. * pi) ** .5) * exp(-0.5 * (mu / sigma) ** 2) + .5 * mu * a
    sigmar = ((mu - mur) * mur + .5 * sigma ** 2 * a) ** .5

    return (mur, sigmar)

def inv_rectified_gaussian(mur, sigmar):
    '''
    Inverse of the function rectified_gaussian
    '''
    if sigmar == 0 * sigmar: # for unit consistency
        return (mur, sigmar)

    x0 = mur / sigmar
    ratio = lambda u, v:u / v
    f = lambda x:ratio(*rectified_gaussian(x, 1.)) - x0
    y = newton(f, x0 * 1.1) # Secant method
    sigma = mur / (exp(-0.5 * y ** 2) / ((2. * pi) ** .5) + .5 * y * (1. + erf(y * (2 ** (-.5)))))
    mu = y * sigma

    return (mu, sigma)

"""
Doubly stochastic processes (Cox processes).
Good for long correlation time constants.
"""

class HomogeneousCorrelatedSpikeTrains(NeuronGroup):
    '''
    Correlated spike trains with identical rates and homogeneous exponential correlations.
    Uses Cox processes (Ornstein-Uhlenbeck).
    '''
    @check_units(r=hertz, tauc=second)
    def __init__(self, N, r, c, tauc, clock=None):
        '''
        Initialization:
        r = rate (Hz)
        c = total correlation strength (in [0,1])
        tauc = correlation time constant (ms)
        Cross-covariance functions are (c*r/tauc)*exp(-|s|/tauc)
        '''
        self.N = N
        # Correction of mu and sigma
        sigmar = (c * r / (2. * tauc)) ** .5
        mu, sigma = inv_rectified_gaussian(r, sigmar)
        eq = Equations('drate/dt=(mu-rate)/tauc + sigma*xi/tauc**.5 : Hz', mu=mu, tauc=tauc, sigma=sigma)
        NeuronGroup.__init__(self, 1, model=eq, threshold=HomogeneousPoissonThreshold(),
                             clock=clock)
        self._use_next_allowed_spiketime_refractoriness = False
        self.rate = mu
        self.LS = SpikeContainer(1) # Spike storage

    def __len__(self):
        # We need to redefine this because it is not the size of the state matrix
        return self.N

    def __getslice__(self, i, j):
        Q = NeuronGroup.__getslice__(self, i, j) # Is this correct?
        Q.N = j - i
        return Q



class MixtureHomogeneousCorrelatedSpikeTrains(NeuronGroup):
    def __init__(self, N, r, c, cl=None):
        """
        Generates a pool of N homogeneous correlated spike trains with identical
        rates r and correlation c (0 <= c <= 1).
        """
        NeuronGroup.__init__(self, N, model="v : 1")
        self.N = N
        self.r = r
        self.c = c
        if cl is None:
            cl = guessclock()
        self.clock = cl
        self.dt = cl.dt

    # called at each time step, returns the spiking neurons?
    def update(self):
        # Generates a source Poisson spike train with rate r/c
        if rand(1) <= self.r / self.c * self.dt:
            # If there is a source spike, it is copied to each target train with probability c
            spikes = nonzero(rand(self.N) <= self.c)[0]
        else:
            spikes = []
        self.LS.push(spikes)



def decompose_correlation_matrix(C, R):
    '''
    Completes the diagonal of C and finds L such that C=LL^T.
    C is matrix of correlation coefficients with unspecified diagonal.
    R is the rate vector.
    C must be symmetric.
    N.B.: The diagonal of C is modified (with zeros).
    '''
    # 0) Remove diagonal entries and calculate D (var_i(x) is should have magnitude r_i^2)
    D = R ** 2
    C -= diag(diag(C))

    # Completion
    # 1) Calculate D^{-1}C
    L = dot(diag(1. / D), C)

    # 2) Find the smallest eigenvalue
    eigenvals = linalg.eig(L)[0]
    alpha = -min(eigenvals[isreal(eigenvals)])

    # 3) Complete the diagonal with alpha*ri^2
    #alpha=alpha+.01; // avoids bad conditioning problems (uncomment if Cholesky fails)
    C += diag(alpha * D)

    # 4) Calculate a square root (Cholesky is unstable, use singular value decomposition)
    #return linalg.cholesky(C,lower=1)
    U, S, V = linalg.svd(C)
    return dot(dot(U, sqrt(diag(S))), V.T)


class CorrelatedSpikeTrains(NeuronGroup):
    '''
    Correlated spike trains with arbitrary rates and pair-wise exponential correlations.
    Uses Cox processes (Ornstein-Uhlenbeck).
    P.rate is the vector of (time-varying) rates.
    '''
    @check_units(tauc=second)
    def __init__(self, rates, C, tauc, clock=None):
        '''
        Initialization:
        rates = rates (Hz)
        C = correlation matrix
        tauc = correlation time constant (ms)
        Cross-covariance functions are C[i,j]*exp(-|s|/tauc)
        '''
        eq = Equations('''
        rate : Hz
        dy/dt = -y*(1./tauc)+xi/(.5*tauc)**.5 : 1
        ''')
        NeuronGroup.__init__(self, len(rates), model=eq, threshold=PoissonThreshold(), \
                             clock=clock)
        self._R = array(rates)
        self._L = decompose_correlation_matrix(array(C), self._R)

    def update(self):
        # Calculate rates
        self.rate_ = self._R + dot(self._L, self.y_)
        NeuronGroup.update(self)

row_vector = lambda V:V.reshape(1, len(V))
column_vector = lambda V:V.reshape(len(V), 1)

def homogeneous_mixture(R, c):
    '''
    Returns a mixture (nu,P) for a homogeneous synchronization structure:
    C(i,j)=2*c*(R(i)*R(j))/<R>
    '''
    pass

def find_mixture(R, C, iter=10000):
    '''
    Finds a mixture matrix P and source rate vector nu
    from the correlation matrix C and the target rates R.
    Returns nu,P.
    
    Gradient descent.
    TODO: use Scipy optimization algorithms.
    '''
    N = len(R)

    # Steps
    b = 0.1 / N
    a = b / N

    # Initial value: here such that F=0
    P = eye(N)
    R = column_vector(R)
    nu = row_vector(R)

    for _ in xrange(iter):
        # Compute error
        Q = dot(P, diag(sqrt(nu.flatten())))
        C2 = dot(Q, Q.T)
        E = linalg.norm(C - C2 - diag(diag(C - C2)), 'fro')
        F = sum(clip(dot(P, nu.T) - R, 0, Inf))
        A = -(C - C2 - diag(diag(C - C2)))

        #print "E=",E,"F=",F,"Etot=",E*a+F*b

        # Gradient in E
        dPE = 4 * dot(dot(A, P), diag(nu.flatten()))
        dNuE = row_vector(2 * diag(dot(dot(P.T, A), P)))

        # Gradient in F
        HF = (dot(P, nu.T) - R) > 0
        dNuF = dot(HF.T, P)
        dPF = dot(HF, nu)

        # One step of gradient descent
        P = P - a * dPE - b * dPF
        nu = nu - a * dNuE - b * dNuF

        # Clipping
        nu = clip(nu, 0, Inf)
        P = clip(P, 0, 1)

    # Now we complete
    nu = hstack((nu, (R - dot(P, nu.T)).T))
    P = hstack((P, eye(N)))
    print E, F

    return nu, P

def mixture_process(nu, P, tauc, t):
    '''
    Generate correlated spike trains from a mixture process.
    nu = rates of source spike trains
    P = mixture matrix
    tauc = correlation time constant
    t = duration
    Returns a list of (neuron_number,spike_time) to be passed to SpikeGeneratorGroup.
    '''
    n = array(poisson(nu * t)) # number of spikes for each source spike train
    if n.ndim == 0:
        n = array([n])
    # Only non-zero entries:
    nonzero = n.nonzero()[0]
    n = n[nonzero]
    P = array(P.take(nonzero, axis=1))
    nik = binomial(n, P) # number of spikes from k in i
    result = []
    for k in xrange(P.shape[1]):
        spikes = rand(n[k]) * t
        for i in xrange(P.shape[0]):
            m = nik[i, k]
            if m > 0:
                if tauc > 0:
                    selection = sample(spikes, m) + array(exponential(tauc, m))
                else:
                    selection = sample(spikes, m)
                result.extend(zip([i] * m, selection))
    result = [(i,t*second) for i,t in result]
    return result

if __name__ == '__main__':
    from time import time
    R = 2 + rand(5)
    C = rand(5, 5)
    C = .5 * (C + C.T)
    t1 = time()
    nu, P = find_mixture(R, C)
    t2 = time()
    print t2 - t1
