'''
This example implements a band pass filter whose center frequency is modulated by
a Ornstein-Uhlenbeck. The white noise term used for this process is output by a  FunctionFilterbank. 
The bandpass filter coefficients update is an example of how to use a ControlFilterbank.
The bandpass filter is a basic biquadratic filter for which the Q factor and the center
frequency must be given. The input is a white noise.
'''

from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *


samplerate=20*kHz #sample frequency of the sound
SoundDuration=300*ms    #duration of the sound
sound=whitenoise(SoundDuration,samplerate=samplerate).ramp()  #generate a white noise 

nchannels=1   #number of frequency channel (here it must be one as a spectrogram of the output is plotted)

fc_init=5000*Hz   #initial center frequency of the band pass filter
Q=5               #quality factor of the band pass filter
update_interval=4        # the filter coefficients are updated every update_interval (here in sample)

#parameters of the Ornstein-Uhlenbeck process
s_i=1200*Hz
tau_i=100*ms      
mu_i=fc_init/tau_i
sigma_i=sqrt(2)*s_i/sqrt(tau_i)
deltaT=defaultclock.dt

#this function  is used in a FunctionFilterbank. It outputs a noise term that will be later used
# by the controler to update the center frequency
noise=lambda x : mu_i*deltaT+sigma_i*randn(1)*sqrt(deltaT)
noise_generator=FunctionFilterbank(sound, noise)

#this class will take as input the output of the noise generator and as target the bandpass filter center frequency
class CoefController:
    def __init__(self,target,samplerate,fc_init,Q,tau_i):
        self.target=target
        self.samplerate=samplerate
        self.deltaT=1./samplerate
        self.tau_i=tau_i
        self.BW=2*arcsinh(1./2/Q)*1.44269
        self.fc=fc_init
        
    def __call__(self,input):
        
        noise_term=input[-1,:]#the  control variables are taken as the last of the buffer
        self.fc=self.fc-self.fc/self.tau_i*self.deltaT+noise_term #update the center frequency by updateing the OU process
        
        w0=2*pi*self.fc/self.samplerate
        #update the coefficient of the biquadratic filterbank
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
updater = CoefController(bandpass_filter,samplerate,fc_init,Q,tau_i)   #the updater

#the controller. Remember it must be the last of the chain
control = ControlFilterbank(bandpass_filter, noise_generator, bandpass_filter, updater, update_interval)          

time_varying_filter_mon=control.buffer_fetch(0, len(sound)) #simulation (on the controler)


figure(1)
pxx, freqs, bins, im = specgram(squeeze(time_varying_filter_mon), NFFT=256, Fs=samplerate, noverlap=240) 
imshow(flipud(pxx),aspect='auto')

show()
