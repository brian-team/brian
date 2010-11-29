from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *
from scipy.io import savemat
from scipy.signal import zpk2tf,bilinear
from time import time

dBlevel=60  # dB level in rms dB SPL
#sound=Sound.load('/home/bertrand/Data/Toolboxes/AIM2006-1.40/Sounds/aimmat.wav')
#samplerate=sound.samplerate
#sound=sound.atintensity(dBlevel)
#sound.samplerate=samplerate

#print 'fs=',sound.samplerate,'duration=',len(sound)/sound.samplerate

#simulation_duration=len(sound)/sound.samplerate 

simulation_duration=50*ms
samplerate=50*kHz
sound = whitenoise(simulation_duration,samplerate).ramp()
#simulation_duration=50*ms+0*ms
data=dict()
data['input']=sound
savemat('/home/bertrand/Data/MatlabProg/AuditoryFilters/noise.mat',data)

nbr_cf=1
cf=erbspace(100*Hz,1000*Hz, nbr_cf) 
cf=atleast_1d(1000.)#log_space(500*Hz, 1000*Hz, nbr_cf)#atleast_1d(1000)#

interval_change=len(sound)
print interval_change


    
class BP_control_update: 
    def __init__(self, fs,cf,control):
        self.fs=fs
        self.cf=atleast_1d(cf)
        self.N=len(self.cf)
        self.iteration=0
        self.control=control
        self.control.buffer_init()
        x_cf=11.9*log10(0.8+cf/456)

        self.f_shift=(10**((x_cf+1.2)/11.9)-0.8)*456-cf

        self.wbw=cf/4
        
        self.filt_b=zeros((len(self.cf), 3, 3))
        self.filt_a=zeros((len(self.cf), 3, 3))
        self.poles=zeros((len(self.cf),3),dtype='complex')

        self.poles[:,0:3]=tile(-2*pi*self.wbw+1j*2*pi*(self.cf+self.f_shift),[3,1]).T  
        self.poles=(1+self.poles/(2*fs))/(1-self.poles/(2*fs))
        self.zeroa=1
        
        self.bfp=2*pi*self.cf
        gain_norm=1
        zz=exp(1j*self.bfp/fs)

        gain_norm=abs((zz**2+2*zz+1)/(zz**2-zz*(self.poles[:,0]+conj(self.poles[:,0]))+self.poles[:,0]*conj(self.poles[:,0])))**3*2

        self.filt_b[:,:,0]=vstack([ones(self.N),2*ones(self.N),ones(self.N)]).T
        self.filt_b[:,:,1]=vstack([ones(self.N),2*ones(self.N),ones(self.N)]).T
        self.filt_b[:,:,2]=vstack([ones(self.N),zeros(self.N),-ones(self.N)]).T
        
        for iorder in xrange(3):
            self.filt_a[:,:,iorder]=vstack([ones(self.N),real(-(squeeze(self.poles[:,iorder])+conj(squeeze(self.poles[:,iorder])))),real(squeeze(self.poles[:,iorder])*conj(squeeze(self.poles[:,iorder])))]).T
        self.filt_b[:,:,2]=self.filt_b[:,:,2]/gain_norm


   

    def __call__(self):
        
         self.buffer_start += self.sub_buffer_length
         control_signal=zeros((1,self.N))#self.control.buffer_fetch(self.buffer_start, self.buffer_start+self.sub_buffer_length)
         t1=time()  
#         wbw=-(real(self.poles[:,0] -control_signal[-1,:]))/2.0/pi
#
#         self.poles[:,0:3]=tile(-2*pi*self.wbw+1j*2*pi*(self.cf+self.f_shift),[3,1]).T          
#         self.poles=(1+self.poles/(2*self.fs))/(1-self.poles/(2*self.fs))
#         for iorder in xrange(3):
#             self.filt_a[:,:,iorder]=vstack([ones(self.N),real(-(squeeze(self.poles[:,iorder])+conj(squeeze(self.poles[:,iorder])))),real(squeeze(self.poles[:,iorder])*conj(squeeze(self.poles[:,iorder])))]).T

         print time()-t1

##### Control Path ####
#### low pass filter for feedback to control band pass
fc_LP_fb=500*Hz
LP_fb= LowPassFilterbank(sound, nbr_cf,fc_LP_fb)

#### wide band pass control
vary_coeff_BP_control=BP_control_update(samplerate,cf,LP_fb)
BP_control= TimeVaryingIIRFilterbank(sound,interval_change,vary_coeff_BP_control)

##### first non linearity of control path
Acp,Bcp,Ccp=100.,2.5,0.60 
func_NL1_control=lambda x:sign(x)*Bcp*log(1.+Acp*abs(x)**Ccp)
NL1_control=FunctionFilterbank(BP_control,func_NL1_control)

### second non linearity of control path
asym,s0,x1,s1=7.,8.,5.,3. 
shift = 1./(1.+asym)
x0 = s0*log((1.0/shift-1)/(1+exp(x1/s1)))
gain80=10**(log10(cf)*0.5732 + 1.5220)
rgain=  10**( log10(cf)*0.4 + 1.9)
average_control=0.3357
nlgain= (gain80 - rgain)/average_control
func_NL2_control=lambda x:(1.0/(1.0+exp(-(x-x0)/s0)*(1.0+exp(-(x-x1)/s1)))-shift)*nlgain
NL2_control=FunctionFilterbank(NL1_control,func_NL2_control)
#

#### control low pass filter
fc_LP_control=800*Hz
#LP_control= LowPassFilterbank(NL2_control, nbr_cf,fc_LP_control)
LP_control=ButterworthFilterbank(NL2_control, nbr_cf, 3, fc_LP_control, btype='low')





    
class BP_signal_update: 
    def __init__(self, fs,cf,control):
        self.fs=fs
        self.cf=atleast_1d(cf)
        self.N=len(self.cf)
        self.control=control
        self.control.buffer_init()
        
        self.rgain=10**(log10(cf)*0.4 + 1.9)
        self.fp1=1.0854*cf-106.0034
        self.ta=10**(log10(cf)*1.0230 + 0.1607)
        self.tb=10**(log10(cf)*1.4292 - 1.1550) - 1000
        
        self.filt_b=zeros((len(self.fp1), 3, 10))
        self.filt_a=zeros((len(self.fp1), 3, 10))
        self.poles=zeros((len(self.fp1),10),dtype='complex')
        
        a=-self.rgain+1j*self.fp1*2*pi

        
        self.poles[:,0:4]=tile(-self.rgain+1j*self.fp1*2*pi,[4,1]).T
        self.poles[:,4:8]=tile(real(self.poles[:,0])- self.ta+1j*(imag(self.poles[:,0])- self.tb),[4,1]).T
        self.poles[:,8:10]=tile((real(self.poles[:,0])+real(self.poles[:,4]))*.5+1j*(imag(self.poles[:,0])+imag(self.poles[:,4]))*.5,[2,1]).T
        self.zeroa=array(-10**( log10(cf)*1.5-0.9 ))

        self.poles=(1+self.poles/(2*fs))/(1-self.poles/(2*fs))
        self.zeroa=(1+self.zeroa/(2*fs))/(1-self.zeroa/(2*fs))
        
        self.bfp=2*pi*self.cf
        gain_norm=1
        zz=exp(1j*self.bfp/fs)
        for ii in xrange(10):
            gain_norm=gain_norm*abs((zz**2-zz*(-1+self.zeroa)-self.zeroa)/(zz**2-zz*(self.poles[:,ii]+conj(self.poles[:,ii]))+self.poles[:,ii]*conj(self.poles[:,ii])))

        for iorder in xrange(10):
            self.filt_b[:,:,iorder]=vstack([ones(len(self.cf)),-(-1+self.zeroa),-self.zeroa]).T
            self.filt_a[:,:,iorder]=vstack([ones(self.N),real(-(squeeze(self.poles[:,iorder])+conj(squeeze(self.poles[:,iorder])))),real(squeeze(self.poles[:,iorder])*conj(squeeze(self.poles[:,iorder])))]).T
        
        self.filt_b[:,:,9]=self.filt_b[:,:,9]/tile(gain_norm,[3,1]).T
   

    def __call__(self):
        
         self.buffer_start += self.sub_buffer_length
         control_signal=zeros((1,len(self.fp1)))#self.control.buffer_fetch(self.buffer_start, self.buffer_start+self.sub_buffer_length)
         t1=time()
         
         self.poles[:,0:4]=tile(-(self.rgain-control_signal[-1,:])+1j*self.fp1*2*pi,[4,1]).T
         self.poles[:,4:8]=tile(real(self.poles[:,0])- self.ta+1j*(imag(self.poles[:,0])- self.tb),[4,1]).T
         self.poles[:,8:10]=tile((real(self.poles[:,0])+real(self.poles[:,4]))*.5+1j*(imag(self.poles[:,0])+imag(self.poles[:,4]))*.5,[2,1]).T
         self.poles=(1+self.poles/(2*self.fs))/(1-self.poles/(2*self.fs))
         for iorder in xrange(10):
            self.filt_a[:,:,iorder]=vstack([ones(self.N),real(-(squeeze(self.poles[:,iorder])+conj(squeeze(self.poles[:,iorder])))),real(squeeze(self.poles[:,iorder])*conj(squeeze(self.poles[:,iorder])))]).T
         print time()-t1

#### Signal Path ####
vary_coeff_BP_signal=BP_signal_update(samplerate,cf,LP_control)
BP_signal= TimeVaryingIIRFilterbank(sound,interval_change,vary_coeff_BP_signal)


#NL2_control.buffer_init()
#signal=NL2_control.buffer_fetch(0, len(sound))

#BP_control.buffer_init()
#signal=BP_control.buffer_fetch(0, len(sound))

LP_control.buffer_init()
signal=LP_control.buffer_fetch(0, len(sound))
t1=time()
#signal=signal_path.buffer_fetch(0, len(sound))
print 'the simulation took',time()-t1,' seconds to run'
#signal=vary_filter_coeff.level_dB[:,:len(sound)].T
#print signal.shape
data=dict()
data['out']=signal.T
savemat('/home/bertrand/Data/MatlabProg/AuditoryFilters/pmfr_BH.mat',data)

figure()
plot(signal)
#imshow(flipud(signal.T),aspect='auto')    
show()