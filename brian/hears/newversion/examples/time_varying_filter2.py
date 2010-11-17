from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *
import scipy.signal as signal
from time import time

samplerate=20*kHz

defaultclock.dt =1./samplerate#20*usecond  
SoundDuration=300*ms    #duration of the whole sequence
sound=whitenoise(SoundDuration,samplerate).ramp() 

nbr_channel=1
########"
fc_init=4000*Hz
s_i=700*Hz
coeff=0.2  #1/Q
tau_i=100*ms
mu_i=fc_init/tau_i
sigma_i=sqrt(2)*s_i/sqrt(tau_i)
weights=array([0,1])
sigma_noise=50*pA
current_bounds=2000*pA
deltaT=defaultclock.dt
interval_change=1

oup=lambda x : mu_i*deltaT+sigma_i*randn(1)*sqrt(deltaT)

mean_center_freq=4*kHz
amplitude=1000*Hz
frequency=10*Hz


#class CenterFrequencyGenerator:
#    def __init__(self,fs,mean_center_freq,amplitude,frequency): 
#        self.fs=fs
#        self.mean_center_freq=mean_center_freq
#        self.amplitude=amplitude
#        self.frequency=frequency 
#        self.t=0*second
#        self.fc=mean_center_freq
#    
#    def __call__(self,input):
#        self.fc=self.mean_center_freq+self.amplitude*sin(2*pi*self.frequency*self.t)
#        self.t=self.t+1./self.fs
#        return self.fc
 
#center_frequency=CenterFrequencyGenerator(samplerate,mean_center_freq,amplitude,frequency)      

center_frequency=lambda x:4000*ones_like(x)
fc_generator=FunctionFilterbank(sound, center_frequency)



class FilterCoeffUpdate:
    def __init__(self, fs,nbr_channel,fc_init,fc_vary,coeff):

        self.filt_b=zeros((nbr_channel, 3, 1))
        self.filt_a=zeros((nbr_channel, 3, 1))
        fc_vary.buffer_init()
        self.fc_vary=fc_vary
        self.fs=fs
        Q=1./coeff
        w0=2*pi*fc_init/self.fs
        self.BW=2*arcsinh(1./2/Q)*1.44269
        alpha=sin(w0)*sinh(log(2)/2*self.BW*w0/sin(w0))
        
        self.filt_b[:, 0, 0]=sin(w0)/2
        self.filt_b[:, 1, 0]=0
        self.filt_b[:, 2, 0]=-sin(w0)/2
    
        self.filt_a[:, 0, 0]=1+alpha
        self.filt_a[:, 1, 0]=-2*cos(w0)
        self.filt_a[:, 2, 0]=1-alpha
  
    def __call__(self):
        self.buffer_start += self.sub_buffer_length
        fc=self.fc_vary.buffer_fetch(self.buffer_start, self.buffer_start+self.sub_buffer_length)
        fc=array(atleast_1d(fc))
        w0=2*pi*fc[-1]/array(self.fs)
     
        alpha=sin(w0)*sinh(log(2)/2*self.BW*w0/sin(w0))
        self.filt_b[:, 0, 0]=sin(w0)/2
        self.filt_b[:, 1, 0]=0
        self.filt_b[:, 2, 0]=-sin(w0)/2
    
        self.filt_a[:, 0, 0]=1+alpha
        self.filt_a[:, 1, 0]=-2*cos(w0)
        self.filt_a[:, 2, 0]=1-alpha

FilterCoeffUpdate_class=FilterCoeffUpdate(samplerate,nbr_channel,fc_init,fc_generator,coeff)
G2= TimeVaryingIIRFilterbank(sound,interval_change,FilterCoeffUpdate_class)


G2.buffer_init()
t1=time()
G2_mon=G2.buffer_fetch(0, len(sound))
print 'the simulation took',time()-t1,' seconds to run'

figure(1)
#plot(G2_mon)
pxx, freqs, bins, im = specgram(squeeze(G2_mon), NFFT=256, Fs=samplerate, noverlap=240) #pylab functio
imshow(pxx,aspect='auto')

show()

show()