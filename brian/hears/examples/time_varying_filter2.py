
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
oup_generatorFB=FunctionFilterbank(sound.rate, nbr_channel, oup)
oup_generator = FilterbankGroup(oup_generatorFB, sound)

#fb= TimeVaryingIIRFilterbank(sound.rate,coeff*ones(nbr_channel),m_i*ones(nbr_channel),s_i*ones(nbr_channel),tau_i*ones(nbr_channel))
#G = FilterbankGroup(fb, sound)
#G_mon = StateMonitor(G, 'output', record=True)
########
#def vary_filter(fs,N,b,a):
#    return b,a

def vary_filter_init(fs,coeff,tau_i,oup):
    N=len(tau_i)
    b=zeros((N, 3, 1))
    a=zeros((N, 3, 1))
    fc=m_i
    deltaT=1./fs
    Q=1./coeff
    BW=2*arcsinh(1./2/Q)*1.44269
    
    
    fc=fc-fc/tau_i*deltaT+oup
    BWhz=fc/Q
    

#    if fc<=50*Hz:
#        fc=50*Hz
#
#    if fc+BWhz/2>=fs/2:
#        fc=fs/2-1000*Hz

    w0=2*pi*fc/fs
    alpha=sin(w0)*sinh(log(2)/2*BW*w0/sin(w0))
    
    b[:, 0, 0]=sin(w0)/2
    b[:, 1, 0]=0
    b[:, 2, 0]=-sin(w0)/2

    a[:, 0, 0]=1+alpha
    a[:, 1, 0]=-2*cos(w0)
    a[:, 2, 0]=1-alpha
    return b,a,[fc,oup,tau_i,deltaT,sigma_i,BW,Q]
      
def vary_filter(fs,N,fc,oup,tau_i,deltaT,sigma_i,BW,Q):

    b=zeros((N, 3, 1))
    a=zeros((N, 3, 1))
    
    fc[:]=fc-fc/tau_i*deltaT+oup
    #print fc,tau_i,deltaT
    BWhz=fc/Q

#    if fc<=50*Hz:
#        fc=50*Hz
#
#    if fc+BWhz/2>=fs/2:
#        fc=fs/2-1000*Hz

    w0=2*pi*fc/fs

    alpha=sin(w0)*sinh(log(2)/2*BW*w0/sin(w0))
    b[:, 0, 0]=sin(w0)/2
    b[:, 1, 0]=0
    b[:, 2, 0]=-sin(w0)/2

    a[:, 0, 0]=1+alpha
    a[:, 1, 0]=-2*cos(w0)
    a[:, 2, 0]=1-alpha
    return b,a


fb2= TimeVaryingIIRFilterbank2(sound.rate,nbr_channel,vary_filter_init,vary_filter,coeff*ones(nbr_channel),tau_i*ones(nbr_channel),oup_generator.output)
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