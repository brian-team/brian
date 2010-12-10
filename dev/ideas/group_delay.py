# -*- coding:utf-8 -*-
"""
Group delay of filters.

This gives really noisy outputs. I guess I made a mistake.
"""
from brian import *
from brian.hears.newversion import *
import os
from scipy.signal import freqz

def group_delay(f):
    '''
    Group delay of a filter.
    Group delay is -d{angle(w)}/dw.
    Smith algorithm.
    '''
    # First compute the FFT
    x=array(f).flatten()
    #x+=max(x)*(1e-20*rand(len(x))-.5*1e-20) # avoid divisions by 0
    dt=1./f.samplerate
    freq=fftfreq(len(x),dt)
    a=fft(x*arange(len(x)))
    b=fft(x)
    return freq[:len(x)/2],dt*(a/b).real[:len(x)/2] # only positive frequencies

choice='IRCAM'
hrtf_locations = [
        r'D:\HRTF\\' + choice,
        r'/home/bertrand/Data/Measurements/HRTF/' + choice,
        r'C:\Documents and Settings\dan\My Documents\Programming\\' + choice
        ]
found = 0
for path in hrtf_locations:
    if os.path.exists(path):
        found = 1
        break
if found == 0:
    raise IOError('Cannot find IRCAM HRTF location, add to ircam_locations')

ircam = IRCAM_LISTEN(path)
h = ircam.load_subject(1002)
h = h.subset(lambda azim,elev:azim==45 and elev==0)

freq,xL=group_delay(h.hrtf[0].left)
_,xR=group_delay(h.hrtf[0].right)
print mean((xL-xR)[:len(xL)/4])*1e6 # in usec
plot(freq,(xL-xR)*1e6)

#plot(h.hrtf[0].left)
#plot(h.hrtf[0].right)
show()
