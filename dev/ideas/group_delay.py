# -*- coding:utf-8 -*-
"""
Group delay of filters.

This gives really noisy outputs. I added a Gaussian filter.

I have a doubt: is group delay the same as narrow band ITD?
No! It's the characteristic delay!

Actually, group delay computes the characteristic delay, not the ITD.
* I think the ITD is the delay with phase consistent with IPD that
is closest to the group delay. *

[WRONG REMARKS BELOW!]
A simple observation:
* characteristic delay is just the mean ITD in a frequency band
* characteristic phase is?
ITD = -dphi/(2*pi*df) (=CD) (more precisely, group delay)
IPD = IPD(f)-ITD*2*pi*df
CP = IPD(f) - dphi/df * f (phase at 0 freq)
   = IPD(f) + CD *2*pi*f

Other idea:
* linear regression of IPD vs frequency in S1, with frequency windows in
  log or ERB space (e.g. 1/3 octave)
  Trick: nonlinear least square regression on vector (sin(phi),cos(phi)) (same as regression
  in complex plane) (could done recursively)
* could be weighted by power at that frequency (or thresholded, e.g., only
  include 50% best frequencies)
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
    freq=fftfreq(len(x),dt)[:len(x)/2]
    a=fft(x*arange(len(x)))
    b=fft(x)
    grpd=dt*(a/b).real[:len(x)/2] # group delay
    # Selection
    ind=(freq>200) & (freq<3000)
    freq=freq[ind]
    grpd=grpd[ind]
    # Smoothing
    width_dt=20
    filter='gaussian'
    window = {'gaussian': exp(-arange(-2 * width_dt, 2 * width_dt + 1) ** 2 * 1. / (2 * (width_dt) ** 2)),
                'flat': ones(width_dt)}[filter]
    y=convolve(grpd, window * 1. / sum(window), mode='same')
    return freq,y # only positive frequencies

def ITD(fL,fR):
    '''
    ITD vs. frequency using group delays
    
    Doesn't work at all!
    '''
    dt=1/fL.samplerate
    fL=array(fL).flatten()
    fR=array(fR).flatten()
    n=len(fL)
    b=fft(fR)/fft(fL)
    a=fft(arange(n)*ifft(b))
    freq=fftfreq(n,dt)
    return freq[:n/2],dt*(a/b).real[:n/2] # only positive frequencies

def IPD_raw(fL,fR):
    dt=1/fL.samplerate
    fL=array(fL).flatten()
    fR=array(fR).flatten()
    n=len(fL)
    xR,xL=fft(fR)[1:n/2],fft(fL)[1:n/2]
    x=xR/xL
    freq=fftfreq(n,dt)[1:n/2]
    return freq,angle(x)

def IPD(fL,fR,threshold=0.1):
    '''
    IPD vs. frequency (by unwrapping).
    Maybe: we should look at the power in these frequencies and threshold.
    There's a magnitude threshold.
    
    An ITD-based unwrapping could be chosen, i.e. minimize difference with
    ITD expectation.
    '''
    dt=1/fL.samplerate
    fL=array(fL).flatten()
    fR=array(fR).flatten()
    n=len(fL)
    xR,xL=fft(fR)[1:n/2],fft(fL)[1:n/2]
    x=xR/xL
    freq=fftfreq(n,dt)[1:n/2]
    ind=(abs(xR)>threshold) & (abs(xL)>threshold)
    unwrapped=unwrap(angle(x[ind]))
    y=zeros(n/2-1)
    y[ind]=unwrapped
    # Fill blanks (assumes linear phase + constant ITD on boundaries)
    ind=find(ind)
    breaks=find(diff(ind)>1)
    for i in range(len(breaks)):
        start=ind[breaks[i]]+1
        end=ind[breaks[i]+1] # not included
        y[start:end]=linspace(y[start-1],y[end],end-start)
    if ind[0]>0:
        y[:ind[0]]=freq[:ind[0]]*y[ind[0]]/freq[ind[0]]
    if ind[-1]<len(y):
        y[ind[-1]+1:]=freq[ind[-1]+1:]*y[ind[-1]]/freq[ind[-1]]
    #subplot(211)
    #plot(freq,abs(fft(fL)[:n/2]))
    #plot(freq,abs(fft(fR)[:n/2]))
    
    return freq,y

def ITD2(fL,fR):
    '''
    ITD vs. frequency
    '''
    dt=1/fL.samplerate
    fL=array(fL).flatten()
    fR=array(fR).flatten()
    n=len(fL)
    x=fft(fR)/fft(fL)
    freq=fftfreq(len(x),dt)[:n/2]
    ipd=unwrap(angle(x)[:n/2])
    itd=-diff(ipd)/(2*pi*diff(freq))
    return freq[:-1],itd

def ITD3(fL,fR,threshold=0.1):
    freq,ipd=IPD(fL,fR,threshold)
    # Smoothing (ITD rather than IPD is smoothed, i.e., assuming small variations in ITD)
    # Width should be in ERB space
    # Differentiation could be done without the blank filling
    width_dt=10
    filter='flat'
    window = {'gaussian': exp(-arange(-2 * width_dt, 2 * width_dt + 1) ** 2 * 1. / (2 * (width_dt) ** 2)),
                'flat': ones(width_dt)}[filter]
    #ipd=convolve(ipd/(2*pi*freq), window * 1. / sum(window), mode='same')*freq
    ipd=convolve(ipd/(2*pi), window * 1. / sum(window), mode='same')
    # I suppose we need a window?

    #itd=ipd/(2*pi*freq) # actually is that the ITD?? maybe not!
    itd=zeros(len(ipd))
    itd[:-1]=-diff(ipd)/diff(freq)
    itd[-1]=itd[-2]
    y = itd

    return freq,y

choice='IRCAM'
hrtf_locations = [
        r'C:\HRTF\\' + choice,
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
h = h.subset(lambda azim,elev:azim==90 and elev==0)

#freq,xL=group_delay(h.hrtf[0].left)
#_,xR=group_delay(h.hrtf[0].right)
#itd=(xR-xL)
left=h.hrtf[0].left
right=h.hrtf[0].right
freq,itd=ITD3(left,right,threshold=0.05)
_,ipd=IPD(left,right,threshold=0.05)
#ind=(freq>200) & (freq<3000)
#freq=freq[ind]
#itd=itd[ind]
#subplot(211)
#plot(freq,itd*1e6)
subplot(312)
ind=(freq<10000) & (freq>180)
plot(freq[ind],itd[ind]*1e6,'r') # CD
plot(freq[ind],-ipd[ind]/(2*pi*freq[ind])*1e6,'b') # ITD
#ylim(0,1500)
subplot(311)
plot(freq[ind],ipd[ind]) # IPD
subplot(313)
CP = IPD_raw(left,right)[1] + itd *2*pi*freq # sure about this??
hist((((CP[ind]+pi) % (2*pi))-pi)/(2*pi))
#plot(freq[ind],(((CP[ind]+pi) % (2*pi))-pi)/(2*pi)) # CP
#ylim(-0.5,0.5)
#plot(smooth(smooth(smooth(smooth(freq)))),smooth(smooth(smooth(smooth(ITD)))))

#plot(h.hrtf[0].left)
#plot(h.hrtf[0].right)
show()
