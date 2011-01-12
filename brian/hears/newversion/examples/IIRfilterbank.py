'''
Example of the use of the class IIRfilterbank available in the library. 
In this example, a white noise is filtered by a  bank of elliptic bandpass filters which are different for every channels. The centre frequency of 
the filters is linearly taken between 100kHz and 1000kHz and its bandwidth increases linearly with frequency.
'''
from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=True,use_gpu = False)
from brian.hears import *


dBlevel=50*dB  # dB level of the input sound in rms dB SPL
sound=whitenoise(100*ms,samplerate=44*kHz).ramp() #generation of a white noise
sound=sound.atlevel(dBlevel) #set the sound to a certain dB level

nchannels=50
center_frequencies=linspace(200*Hz,1000*Hz, nchannels)  #center frequencies 
bw=linspace(50*Hz,300*Hz, nchannels)
gpass=2.
gstop=10.

passband=vstack((center_frequencies-bw/2,center_frequencies+bw/2)) #maybe better with a list of two array??

stopband=vstack((center_frequencies-1.5*bw,center_frequencies+1.5*bw)) #


gammatone =IIRFilterbank(sound,nchannels, passband, stopband, gpass, gstop, 'bandstop','cheby1') #instantiation of the filterbank

gt_mon=gammatone.buffer_fetch(0, len(sound)) #processing


figure()
imshow(flipud(gt_mon.T),aspect='auto')    
show()




    