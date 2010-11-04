'''
Spike statistics
----------------
In all functions below, spikes is a sorted list of spike times
'''
from numpy import *
from brian.units import check_units, second
from brian.stdunits import ms, Hz
from operator import itemgetter

__all__ = ['firing_rate', 'CV', 'correlogram', 'autocorrelogram', 'CCF', 'ACF', 'CCVF', 'ACVF', 'group_correlations', 'sort_spikes',
         'total_correlation', 'vector_strength', 'gamma_factor', 'get_gamma_factor_matrix', 'get_gamma_factor']

# First-order statistics
def firing_rate(spikes):
    '''
    Rate of the spike train.
    '''
    if spikes==[]:
        return NaN
    return (len(spikes) - 1) / (spikes[-1] - spikes[0])

def CV(spikes):
    '''
    Coefficient of variation.
    '''
    if spikes==[]:
        return NaN
    ISI = diff(spikes) # interspike intervals
    return std(ISI) / mean(ISI)

# Second-order statistics
def correlogram(T1, T2, width=20 * ms, bin=1 * ms, T=None):
    '''
    Returns a cross-correlogram with lag in [-width,width] and given bin size.
    T is the total duration (optional) and should be greater than the duration of T1 and T2.
    The result is in Hz (rate of coincidences in each bin).

    N.B.: units are discarded.
    TODO: optimise?
    '''
    if (T1==[]) or (T2==[]): # empty spike train
        return NaN
    # Remove units
    width = float(width)
    T1 = array(T1)
    T2 = array(T2)
    i = 0
    j = 0
    n = int(ceil(width / bin)) # Histogram length
    l = []
    for t in T1:
        while i < len(T2) and T2[i] < t - width: # other possibility use searchsorted
            i += 1
        while j < len(T2) and T2[j] < t + width:
            j += 1
        l.extend(T2[i:j] - t)
    H, _ = histogram(l, bins=arange(2 * n + 1) * bin - n * bin, new = True)

    # Divide by time to get rate
    if T is None:
        T = max(T1[-1], T2[-1]) - min(T1[0], T2[0])
    # Windowing function (triangle)
    W = zeros(2 * n)
    W[:n] = T - bin * arange(n - 1, -1, -1)
    W[n:] = T - bin * arange(n)

    return H / W

def autocorrelogram(T0, width=20 * ms, bin=1 * ms, T=None):
    '''
    Returns an autocorrelogram with lag in [-width,width] and given bin size.
    T is the total duration (optional) and should be greater than the duration of T1 and T2.
    The result is in Hz (rate of coincidences in each bin).

    N.B.: units are discarded.
    '''
    return correlogram(T0, T0, width, bin, T)

def CCF(T1, T2, width=20 * ms, bin=1 * ms, T=None):
    '''
    Returns the cross-correlation function with lag in [-width,width] and given bin size.
    T is the total duration (optional).
    The result is in Hz**2:
    CCF(T1,T2)=<T1(t)T2(t+s)>

    N.B.: units are discarded.
    '''
    return correlogram(T1, T2, width, bin, T) / bin

def ACF(T0, width=20 * ms, bin=1 * ms, T=None):
    '''
    Returns the autocorrelation function with lag in [-width,width] and given bin size.
    T is the total duration (optional).
    The result is in Hz**2:
    ACF(T0)=<T0(t)T0(t+s)>

    N.B.: units are discarded.
    '''
    return CCF(T0, T0, width, bin, T)

def CCVF(T1, T2, width=20 * ms, bin=1 * ms, T=None):
    '''
    Returns the cross-covariance function with lag in [-width,width] and given bin size.
    T is the total duration (optional).
    The result is in Hz**2:
    CCVF(T1,T2)=<T1(t)T2(t+s)>-<T1><T2>

    N.B.: units are discarded.
    '''
    return CCF(T1, T2, width, bin, T) - firing_rate(T1) * firing_rate(T2)

def ACVF(T0, width=20 * ms, bin=1 * ms, T=None):
    '''
    Returns the autocovariance function with lag in [-width,width] and given bin size.
    T is the total duration (optional).
    The result is in Hz**2:
    ACVF(T0)=<T0(t)T0(t+s)>-<T0>**2

    N.B.: units are discarded.
    '''
    return CCVF(T0, T0, width, bin, T)

def total_correlation(T1, T2, width=20 * ms, T=None):
    '''
    Returns the total correlation coefficient with lag in [-width,width].
    T is the total duration (optional).
    The result is a real (typically in [0,1]):
    total_correlation(T1,T2)=int(CCVF(T1,T2))/rate(T1)
    '''
    if (T1==[]) or (T2==[]): # empty spike train
        return NaN
    # Remove units
    width = float(width)
    T1 = array(T1)
    T2 = array(T2)
    # Divide by time to get rate
    if T is None:
        T = max(T1[-1], T2[-1]) - min(T1[0], T2[0])
    i = 0
    j = 0
    x = 0
    for t in T1:
        while i < len(T2) and T2[i] < t - width: # other possibility use searchsorted
            i += 1
        while j < len(T2) and T2[j] < t + width:
            j += 1
        x += sum(1. / (T - abs(T2[i:j] - t))) # counts coincidences with windowing (probabilities)
    return float(x / firing_rate(T1)) - float(firing_rate(T2) * 2 * width)

def sort_spikes(spikes):
    """
    Sorts spikes stored in a (i,t) list by time.
    """
    spikes = sorted(spikes, key=itemgetter(1))
    return spikes

def group_correlations(spikes, delta=None):
    """
    Computes the pairwise correlation strength and timescale of the given pool of spike trains.
    spikes is a (i,t) list and must be sorted.
    delta is the length of the time window, 10*ms by default.
    """
    aspikes = array(spikes)
    N = aspikes[:, 0].max() + 1 # neuron count
    T = aspikes[:, 1].max() # total duration
    spikecount = zeros(N)
    tauc = zeros((N, N))
    S = zeros((N, N))
    if delta is None:
        delta = 10 * ms # size of the window
    windows = -2 * delta * ones(N) # windows[i] is the end of the window for neuron i = lastspike[i}+delta
    for i, t in spikes:
        sources = (t <= windows) # neurons such that (i,t) is a target spike for them
        if sum(sources) > 0:
            indices = nonzero(sources)[0]
            S[indices, i] += 1
            delays = t - windows[indices] + delta
#            print i, t, indices, delays
            tauc[indices, i] += delays
        spikecount[i] += 1
        windows[i] = t + delta # window update

    tauc /= S

    S = S / tile(spikecount.reshape((-1, 1)), (1, N)) # normalize S
    rates = spikecount / T
    S = S - tile(rates.reshape((1, -1)), (N, 1)) * delta

    S[isnan(S)] = 0.0
    tauc[isnan(tauc)] = 0.0

    return S, tauc

# Phase-locking properties
def vector_strength(spikes, period):
    '''
    Returns the vector strength of the given train
    '''
    return abs(mean(exp(array(spikes) * 1j * 2 * pi / period)))

# Normalize the coincidence count of two spike trains (return the gamma factor)
def get_gamma_factor(coincidence_count, model_length, target_length, target_rates, delta):
    NCoincAvg = 2 * delta * target_length * target_rates
    norm = .5 * (1 - 2 * delta * target_rates)
    gamma = (coincidence_count - NCoincAvg) / (norm * (target_length + model_length))
    return gamma

# Normalize the coincidence matrix between a set of  trains (return the gamma factor matrix)
def get_gamma_factor_matrix(coincidence_matrix, model_length, target_length, target_rates, delta):

    target_lengthMAT =tile(target_length,(len(model_length),1))
    target_rateMAT =tile(target_rates,(len(model_length),1))
    model_lengthMAT  =tile(model_length.reshape(-1,1),(1,len(target_length)))
    NCoincAvg  =2 * delta * target_lengthMAT * target_rateMAT 
    norm =.5 * (1 - 2 * delta * target_rateMAT)

   # print  target_rateMAT 
    print coincidence_matrix 
    #print NCoincAvg
    #print (norm * (target_lengthMAT + model_lengthMAT))
    gamma = (coincidence_matrix - NCoincAvg) / (norm * (target_lengthMAT + model_lengthMAT))
    gamma=triu(gamma,0)+triu(gamma,1).T
    return gamma


# Gamma factor
@check_units(delta=second)
def gamma_factor(source, target, delta, normalize=True, dt=None):
    '''
    Returns the gamma precision factor between source and target trains,
    with precision delta.
    source and target are lists of spike times.
    If normalize is True, the function returns the normalized gamma factor 
    (less than 1.0), otherwise it returns the number of coincidences.
    dt is the precision of the trains, by default it is defaultclock.dt
    
    Reference:
    R. Jolivet et al., 'A benchmark test for a quantitative assessment of simple neuron models',
        Journal of Neuroscience Methods 169, no. 2 (2008): 417-424.
    '''

    source = array(source)
    target = array(target)
    target_rate = firing_rate(target) * Hz

    if dt is None:
        delta_diff = delta
    else:
        source = array(rint(source / dt), dtype=int)
        target = array(rint(target / dt), dtype=int)
        delta_diff = int(rint(delta / dt))

    source_length = len(source)
    target_length = len(target)

    if (target_length == 0 or source_length == 0):
        return 0

    if (source_length > 1):
        bins = .5 * (source[1:] + source[:-1])
        indices = digitize(target, bins)
        diff = abs(target - source[indices])
        matched_spikes = (diff <= delta_diff)
        coincidences = sum(matched_spikes)
    else:
        indices = [amin(abs(source - target[i])) <= delta_diff for i in xrange(target_length)]
        coincidences = sum(indices)

    # Normalization of the coincidences count
#    NCoincAvg = 2 * delta * target_length * target_rate
#    norm = .5*(1 - 2 * target_rate * delta)
#    gamma = (coincidences - NCoincAvg)/(norm*(source_length + target_length))

    # TODO: test this
    gamma = get_gamma_factor(coincidences, source_length, target_length, target_rate, delta)

    if normalize:
        return gamma
    else:
        return coincidences

if __name__ == '__main__':

    from brian import *

    print vector_strength([1.1 * ms, 1 * ms, .9 * ms], 2 * ms)

    N = 100000
    T1 = cumsum(rand(N) * 10 * ms)
    T2 = cumsum(rand(N) * 10 * ms)
    duration = T1[N / 2] # Cut so that both spike trains have the same duration
    T1 = T1[T1 < duration]
    T2 = T2[T2 < duration]
    print firing_rate(T1)
    C = CCVF(T1, T2, bin=1 * ms)
    print total_correlation(T1, T2)
    plot(C)
    show()
