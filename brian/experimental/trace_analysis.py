'''
Analysis of voltage traces (current-clamp experiments or HH models).
'''
from numpy import *

def spike_times(v):
    '''
    Returns the indexes of spike times.
    '''
    return ((v[1:]>0) & (v[:-1]<0)).nonzero()[0]
