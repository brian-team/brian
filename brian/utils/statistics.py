'''
Spike statistics
----------------
In all functions below, spikes is a sorted list of spike times
'''
from numpy import *
from brian.units import check_units,second
from brian.stdunits import ms,Hz

__all__=['firing_rate','CV','correlogram','autocorrelogram','CCF','ACF','CCVF','ACVF',
         'total_correlation','vector_strength','gamma_factor']

# First-order statistics
def firing_rate(spikes):
    '''
    Rate of the spike train.
    '''
    return (len(spikes)-1)/(spikes[-1]-spikes[0])

def CV(spikes):
    '''
    Coefficient of variation.
    '''
    ISI=diff(spikes) # interspike intervals
    return std(ISI)/mean(ISI)

# Second-order statistics
def correlogram(T1,T2,width=20*ms,bin=1*ms,T=None):
    '''
    Returns a cross-correlogram with lag in [-width,width] and given bin size.
    T is the total duration (optional) and should be greater than the duration of T1 and T2.
    The result is in Hz (rate of coincidences in each bin).

    N.B.: units are discarded.
    TODO: optimise?
    '''
    # Remove units
    width=float(width)
    T1=array(T1)
    T2=array(T2)
    i=0
    j=0
    n=int(ceil(width/bin)) # Histogram length
    l=[]
    for t in T1:
        while i<len(T2) and T2[i]<t-width: # other possibility use searchsorted
            i+=1
        while j<len(T2) and T2[j]<t+width:
            j+=1
        l.extend(T2[i:j]-t)
    H,_=histogram(l,bins=arange(2*n+1)*bin-n*bin)
    
    # Divide by time to get rate
    if T is None:
        T=max(T1[-1],T2[-1])-min(T1[0],T2[0])
    # Windowing function (triangle)
    W=zeros(2*n)
    W[:n]=T-bin*arange(n-1,-1,-1)
    W[n:]=T-bin*arange(n)
    
    return H/W

def autocorrelogram(T0,width=20*ms,bin=1*ms,T=None):
    '''
    Returns an autocorrelogram with lag in [-width,width] and given bin size.
    T is the total duration (optional) and should be greater than the duration of T1 and T2.
    The result is in Hz (rate of coincidences in each bin).

    N.B.: units are discarded.
    '''
    return correlogram(T0,T0,width,bin,T)

def CCF(T1,T2,width=20*ms,bin=1*ms,T=None):
    '''
    Returns the cross-correlation function with lag in [-width,width] and given bin size.
    T is the total duration (optional).
    The result is in Hz**2:
    CCF(T1,T2)=<T1(t)T2(t+s)>

    N.B.: units are discarded.
    '''
    return correlogram(T1,T2,width,bin,T)/bin

def ACF(T0,width=20*ms,bin=1*ms,T=None):
    '''
    Returns the autocorrelation function with lag in [-width,width] and given bin size.
    T is the total duration (optional).
    The result is in Hz**2:
    ACF(T0)=<T0(t)T0(t+s)>

    N.B.: units are discarded.
    '''
    return CCF(T0,T0,width,bin,T)

def CCVF(T1,T2,width=20*ms,bin=1*ms,T=None):
    '''
    Returns the cross-covariance function with lag in [-width,width] and given bin size.
    T is the total duration (optional).
    The result is in Hz**2:
    CCVF(T1,T2)=<T1(t)T2(t+s)>-<T1><T2>

    N.B.: units are discarded.
    '''
    return CCF(T1,T2,width,bin,T)-firing_rate(T1)*firing_rate(T2)

def ACVF(T0,width=20*ms,bin=1*ms,T=None):
    '''
    Returns the autocovariance function with lag in [-width,width] and given bin size.
    T is the total duration (optional).
    The result is in Hz**2:
    ACVF(T0)=<T0(t)T0(t+s)>-<T0>**2

    N.B.: units are discarded.
    '''
    return CCVF(T0,T0,width,bin,T)

def total_correlation(T1,T2,width=20*ms,T=None):
    '''
    Returns the total correlation coefficient with lag in [-width,width].
    T is the total duration (optional).
    The result is a real (typically in [0,1]):
    total_correlation(T1,T2)=int(CCVF(T1,T2))/rate(T1)
    '''
    # Remove units
    width=float(width)
    T1=array(T1)
    T2=array(T2)
    # Divide by time to get rate
    if T is None:
        T=max(T1[-1],T2[-1])-min(T1[0],T2[0])
    i=0
    j=0
    x=0
    for t in T1:
        while i<len(T2) and T2[i]<t-width: # other possibility use searchsorted
            i+=1
        while j<len(T2) and T2[j]<t+width:
            j+=1
        x+=sum(1./(T-abs(T2[i:j]-t))) # counts coincidences with windowing (probabilities)
    return float(x/firing_rate(T1))-float(firing_rate(T2)*2*width)

# Phase-locking properties
def vector_strength(spikes,period):
    '''
    Returns the vector strength of the given train
    '''
    return abs(mean(exp(array(spikes)*1j*2*pi/period)))

# Normalize the coincidence count of two spike trains (return the gamma factor)
def get_gamma_factor(coincidence_count, model_length, target_length, target_rates, delta):
    NCoincAvg = 2 * delta * target_length * target_rates
    norm = .5*(1 - 2 * delta * target_rates)    
    gamma = (coincidence_count - NCoincAvg)/(norm*(target_length + model_length))
    return gamma

# Gamma factor
@check_units(delta=second)
def gamma_factor(source, target, delta, normalize = True, dt = None):
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
    target_rate = firing_rate(target)*Hz
    
    if dt is None:
        delta_diff = delta
    else:
        source = array(rint(source/dt), dtype=int)
        target = array(rint(target/dt), dtype=int)
        delta_diff = int(rint(delta/dt))
    
    source_length = len(source)
    target_length = len(target)
    
    if (target_length == 0 or source_length == 0):
        return 0
    
    if (source_length>1):
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

if __name__=='__main__':
    
    from brian import *
    
    print vector_strength([1.1*ms,1*ms,.9*ms],2*ms)
    
    N=100000
    T1=cumsum(rand(N)*10*ms)
    T2=cumsum(rand(N)*10*ms)
    duration=T1[N/2] # Cut so that both spike trains have the same duration
    T1=T1[T1<duration]
    T2=T2[T2<duration]
    print firing_rate(T1)
    C=CCVF(T1,T2,bin=1*ms)
    print total_correlation(T1,T2)
    plot(C)
    show()
