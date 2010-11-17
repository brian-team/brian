
#UNDER DEVELOPMENT!!!

from brian import *
set_global_preferences(useweave=True)
set_global_preferences(weavecompiler ='gcc')
set_global_preferences(usenewpropagate=True)
    
set_global_preferences(usecodegen=True)
set_global_preferences(usecodegenweave=True)
from brian import *
from brian.hears import*
import scipy.signal as signal

samplerate=20*kHz

defaultclock.dt =1./samplerate#20*usecond  
SoundDuration=300*ms    #duration of the whole sequence
sound=whitenoise(SoundDuration,rate=samplerate).ramp() 

nbr_channel=2 
########"
fc_init=4000*Hz
s_i=1000*Hz
coeff=0.2  #1/Q
tau_i=100*ms
mu_i=fc_init/tau_i
sigma_i=sqrt(2)*s_i/sqrt(tau_i)
weights=array([0,1])
sigma_noise=50*pA
current_bounds=2000*pA
deltaT=defaultclock.dt

oup=lambda x : mu_i*deltaT+sigma_i*randn(1)*sqrt(deltaT)
oup_generatorFB=FunctionFilterbank(sound.rate, nbr_channel, oup)
oup_generator = FilterbankGroup(oup_generatorFB, sound)


class FilterCoeffUpdate:
    def __init__(self,fs,coeff,fc_init,tau_i,oup):
        N=len(tau_i)
        self.filt_b=zeros((N, 3, 1))
        self.filt_a=zeros((N, 3, 1))
        
        self.fc=fc_init
        self.fs=fs
        self.deltaT=1./fs
        self.tau_i=tau_i
        self.oup=oup
        Q=1./coeff
        self.BW=2*arcsinh(1./2/Q)*1.44269
    
        w0=2*pi*self.fc/fs
        alpha=sin(w0)*sinh(log(2)/2*self.BW*w0/sin(w0))
        
        self.filt_b[:, 0, 0]=sin(w0)/2
        self.filt_b[:, 1, 0]=0
        self.filt_b[:, 2, 0]=-sin(w0)/2
    
        self.filt_a[:, 0, 0]=1+alpha
        self.filt_a[:, 1, 0]=-2*cos(w0)
        self.filt_a[:, 2, 0]=1-alpha
          
    def __call__(self):
    
        self.fc=self.fc-self.fc/self.tau_i*self.deltaT+self.oup

        w0=2*pi*self.fc/self.fs
    
        alpha=sin(w0)*sinh(log(2)/2*self.BW*w0/sin(w0))
        self.filt_b[:, 0, 0]=sin(w0)/2
        self.filt_b[:, 1, 0]=0
        self.filt_b[:, 2, 0]=-sin(w0)/2
    
        self.filt_a[:, 0, 0]=1+alpha
        self.filt_a[:, 1, 0]=-2*cos(w0)
        self.filt_a[:, 2, 0]=1-alpha

FilterCoeffUpdate_class=FilterCoeffUpdate(sound.rate,coeff,fc_init,tau_i*ones((nbr_channel)),oup_generator.output)
fb2= TimeVaryingIIRFilterbank2(sound.rate,nbr_channel,FilterCoeffUpdate_class)

G2 = FilterbankGroup(fb2, sound)
G2_mon = StateMonitor(G2, 'output', record=True)

run(SoundDuration,report='text')

#G_mon.plot()
#print G2_mon[0]
#figure(3)
#plot(G2_mon[0])

figure(1)
#plot(G2_mon[0])
pxx, freqs, bins, im = specgram(G2_mon[1], NFFT=256, Fs=samplerate, noverlap=240) #pylab functio
imshow(pxx,aspect='auto')

#figure(2)
#pxx, freqs, bins, im = specgram(G2_mon[0], NFFT=256, Fs=samplerate, noverlap=240) #pylab functio
#imshow(pxx,aspect='auto')

show()