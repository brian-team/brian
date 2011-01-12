'''
Example of the use of the class Butterworth available in the library. 
In this example, a white noise is filtered by a  bank of butterworth bandpass filters and lowpass filters which are different for every channels. The centre  or
cutoff frequency of the filters are linearly taken between 100kHz and 1000kHz and its bandwidth  frequency increases linearly with frequency.
'''
from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=True,use_gpu = False)
from brian.hears import *

dBlevel=50*dB  # dB level of the input sound in rms dB SPL
sound=whitenoise(100*ms,samplerate=44*kHz).ramp() #generation of a white noise
sound=sound.atlevel(dBlevel) #set the sound to a certain dB level
order=2 #order of the filters

#### example of a bank of bandpass filter ################
nchannels=50
center_frequencies=linspace(100*Hz,1000*Hz, nchannels)  #center frequencies 
bw=linspace(50*Hz,300*Hz, nchannels)  #bandwidth of the filters

fc=vstack((center_frequencies-bw/2,center_frequencies+bw/2)) #arrays of shape (2 x nchannels) defining the passband frequencies (Hz)

filterbank =Butterworth(sound,nchannels, order, fc, 'bandpass') #instantiation of the filterbank

filterbank_mon=filterbank.buffer_fetch(0, len(sound)) #processing

figure()
subplot(211)
imshow(flipud(filterbank_mon.T),aspect='auto')    


### example of a bank of lowpass filter ################
nchannels=50
cutoff_frequencies=linspace(200*Hz,1000*Hz, nchannels)  #center frequencies 

filterbank =Butterworth(sound,nchannels, order, cutoff_frequencies, 'low') #instantiation of the filterbank

filterbank_mon=filterbank.buffer_fetch(0, len(sound)) #processing


subplot(212)
imshow(flipud(filterbank_mon.T),aspect='auto')    
show()





    