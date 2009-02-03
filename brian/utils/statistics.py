'''
Spike statistics
----------------
In all functions below, spikes is a sorted list of spike times
'''
from numpy import *
from brian.stdunits import ms

__all__=['firing_rate','CV','correlogram','autocorrelogram','CCF','ACF','CCVF','ACVF',
         'total_correlation','vector_strength']

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
    #print std(C)*second