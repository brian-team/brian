
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
sound=whitenoise(SoundDuration,samplerate).ramp() 

nbr_channel=2
########"
m_i=4000
s_i=700
coeff=0.2  #1/Q
tau_i=100*ms
mu_i=m_i/tau_i
sigma_i=sqrt(2)*s_i/sqrt(tau_i)
weights=array([0,1])
sigma_noise=50*pA
current_bounds=2000*pA
deltaT=defaultclock.dt

oup=lambda x : mu_i*deltaT+sigma_i*randn(1)*sqrt(deltaT)

mean_center_freq=4*kHz
amplitude=1000*Hz
frequency=10*Hz

def center_frequency_init(fs,mean_center_freq,amplitude,frequency):  
    t=[0*second]
    return fs,t,mean_center_freq,amplitude,frequency

def center_frequency(input,fs,t,mean_center_freq,amplitude,frequency):
    fc=mean_center_freq+amplitude*sin(2*pi*frequency*t[0])
    t[0]=t[0]+1./fs
    #print t
    return fc
      
fc_generatorFB=FunctionFilterbank(sound.rate, nbr_channel,center_frequency,center_frequency_init,mean_center_freq,amplitude,frequency)
fc_generator = FilterbankGroup(fc_generatorFB, sound)


def vary_filter_init(fs,nbr_channel,fc,coeff):
    N=nbr_channel
    b=zeros((N, 3, 1))
    a=zeros((N, 3, 1))


    w0=2*pi*fc/fs
    Q=1./coeff
    BW=2*arcsinh(1./2/Q)*1.44269
    alpha=sin(w0)*sinh(log(2)/2*BW*w0/sin(w0))
    
    b[:, 0, 0]=sin(w0)/2
    b[:, 1, 0]=0
    b[:, 2, 0]=-sin(w0)/2

    a[:, 0, 0]=1+alpha
    a[:, 1, 0]=-2*cos(w0)
    a[:, 2, 0]=1-alpha
    return b,a,[fc,BW]
      
def vary_filter(fs,N,fc,BW):

    b=zeros((N, 3, 1))
    a=zeros((N, 3, 1))
    
    w0=2*pi*fc/fs

    alpha=sin(w0)*sinh(log(2)/2*BW*w0/sin(w0))
    b[:, 0, 0]=sin(w0)/2
    b[:, 1, 0]=0
    b[:, 2, 0]=-sin(w0)/2

    a[:, 0, 0]=1+alpha
    a[:, 1, 0]=-2*cos(w0)
    a[:, 2, 0]=1-alpha
    return b,a


fb2= TimeVaryingIIRFilterbank2(sound.rate,nbr_channel,vary_filter_init,vary_filter,nbr_channel,fc_generator.output,coeff)
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