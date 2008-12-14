
'''
Spike statistics
----------------
In all functions below, spikes is a sorted list of spike times
'''
from numpy import diff,std,mean,array,dot,ones
from brian.stdunits import ms

# First-order statistics
def rate(spikes):
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
def correlogram(T1,T2,tmax=20*ms):
    '''
    Returns a list of differences of spike times t2-t1 within [-tmax,tmax].
    Does not return a histogram (use numpy histogram function).
    
    TODO: boundary check
    '''
    # Remove units
    tmax=float(tmax)
    T1=array(T1)
    T2=array(T2)
    l=[]
    i=0
    j=0
    for t in T1:
        while i<len(T2) and T2[i]<t-tmax: # other possibility use searchsorted
            i+=1
        while j<len(T2) and T2[j]<t+tmax:
            j+=1
        l.extend(T2[i:j]-t)
        
    return l

if __name__=='__main__':
    from brian import *
    N=1000
    T1=cumsum(rand(N)*10*ms)
    T2=cumsum(rand(N)*10*ms)
    C=correlogram(T1,T2)
    print std(C)*second