'''
This example implements a band pass filter whose center frequency is modulated by
a sinusoid function. This modulator is implemented as a FunctionFilterbank. One 
state variable (here time) must be kept; it is therefore implemented with a class.
The bandpass filter coefficients update is an example of how to use a ControlFilterbank.
The bandpass filter is a basic biquad filter for which the Q factor and the center
frequency must be given. The input is a white noise.
'''

from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *
import scipy.signal as signal
from time import time

samplerate=20*kHz #sample frequency of the sound
defaultclock.dt =1./samplerate 
sound_duration=300*ms    #duration of the sound
sound=whitenoise(sound_duration,samplerate).ramp() #generate a white noise 

nchannels=1       #number of frequency channel (here it must be one as a spectrogram of the output is plotted)

fc_init=5000*Hz   #initial center frequency of the band pass filter
Q=5               #quality factor of the band pass filter
interval=1        #interval (here in sample) every which the filter coefficients are updated

mean_center_freq=4*kHz #mean frequency around which the center frequency will oscillate
amplitude=1500*Hz      #amplitude of the oscillation
frequency=10*Hz        #frequency of the oscillation

#this class is used in a FunctionFilterbank (via its __call__). It outputs the center frequency
#of the band pass filter. Its output is thus later passed as input to the controler. 
class CenterFrequencyGenerator:
    def __init__(self,samplerate,mean_center_freq,amplitude,frequency): 
        self.samplerate=samplerate
        self.mean_center_freq=mean_center_freq
        self.amplitude=amplitude
        self.frequency=frequency 
        self.t=0*second
    
    def __call__(self,input):
        self.fc=self.mean_center_freq+self.amplitude*sin(2*pi*self.frequency*self.t) #update of the center frequency
        self.t=self.t+1./self.samplerate #update of the state variable
        return self.fc

center_frequency=CenterFrequencyGenerator(samplerate,mean_center_freq,amplitude,frequency)      

fc_generator=FunctionFilterbank(sound, center_frequency)

#the updater of the controller generates new filter coefficient of the band pass filter
#based on the center frequency it receives from the fc_generator (its input)
class CoefController:
    def __init__(self, target,Q):
        self.samplerate=target.samplerate
        self.BW=2*arcsinh(1./2/Q)*1.44269
        self.target=target
        
    def __call__(self, input):
        fc=input[-1,:] #the  control variables are taken as the last of the buffer
        w0=2*pi*fc/array(self.samplerate)    
        alpha=sin(w0)*sinh(log(2)/2*self.BW*w0/sin(w0))
        
        self.target.filt_b[:, 0, 0]=sin(w0)/2
        self.target.filt_b[:, 1, 0]=0
        self.target.filt_b[:, 2, 0]=-sin(w0)/2
    
        self.target.filt_a[:, 0, 0]=1+alpha
        self.target.filt_a[:, 1, 0]=-2*cos(w0)
        self.target.filt_a[:, 2, 0]=1-alpha

# In the present example the time varying filter is a LinearFilterbank therefore
#we must initialise the filter coefficients; the one used for the first buffer computation
w0=2*pi*fc_init/samplerate
BW=2*arcsinh(1./2/Q)*1.44269
alpha=sin(w0)*sinh(log(2)/2*BW*w0/sin(w0))

filt_b=zeros((nchannels, 3, 1))
filt_a=zeros((nchannels, 3, 1))

filt_b[:, 0, 0]=sin(w0)/2
filt_b[:, 1, 0]=0
filt_b[:, 2, 0]=-sin(w0)/2

filt_a[:, 0, 0]=1+alpha
filt_a[:, 1, 0]=-2*cos(w0)
filt_a[:, 2, 0]=1-alpha

bandpass_filter = LinearFilterbank(sound,filt_b,filt_a) #the filter which will have time varying coefficients
updater = CoefController(bandpass_filter,Q)   #the updater

#the controler. Remember it must be the last of the chain
control = ControlFilterbank(bandpass_filter, fc_generator, bandpass_filter, updater, interval)   
        

t1=time()
time_varying_filter_mon=control.buffer_fetch(0, len(sound)) #simulation (on the controler)
print 'the simulation took',time()-t1,' seconds to run'

figure(1)
pxx, freqs, bins, im = specgram(squeeze(time_varying_filter_mon), NFFT=256, Fs=samplerate, noverlap=240) #pylab functio
imshow(flipud(pxx),aspect='auto')

show()
