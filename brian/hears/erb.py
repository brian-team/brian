'''
Utility functions adapted from MAP 
'''
from brian import *

__all__ = ['erbspace']

@check_units(low=Hz, high=Hz)
def erbspace(low, high, N, earQ=9.26449, minBW=24.7, order=1):
    '''
    Returns the centre frequencies on an ERB scale.
    
    low, high
        Lower and upper frequencies
    N
        Number of channels
    earQ, minBW, order
        Glasberg and Moore Parameters
    '''
    low = float(low)
    high = float(high)        
    cf = -(earQ*minBW) + exp((arange(1,N+1))*(-log(high + earQ*minBW) + \
            log(low + earQ*minBW))/N) * (high + earQ*minBW)
    cf = cf[::-1]
    return cf

# Testing
if __name__=='__main__':
    cf = erbspace(20*Hz, 20*kHz, 3000)
    plot(cf)
    show()
