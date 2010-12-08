'''

Implementation of the non linear auditory filterbank model from Tan, G. and Carney, L., 
A phenomenological model for the responses of auditory-nerve
fibers. II. Nonlinear tuning with a frequency glide, JASA 2003


The model consists of a control path and a signal path. The control path controls both its own bandwidth via a feedback
loop and also the bandwidth of the signal path. 


'''

from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *
from scipy.io import savemat
from time import time

#dBlevel=60*dB  # dB level in rms dB SPL
#sound=Sound.load('/home/bertrand/Data/Toolboxes/AIM2006-1.40/Sounds/aimmat.wav')
#samplerate=sound.samplerate
#sound=sound.atlevel(dBlevel)
#sound.samplerate=samplerate

#print 'samplerate=',sound.samplerate,'duration=',len(sound)/sound.samplerate

#simulation_duration=len(sound)/sound.samplerate 

simulation_duration=50*ms
samplerate=50*kHz
sound = whitenoise(simulation_duration,samplerate)#.ramp()
simulation_duration=50*ms+0*ms
data=dict()
data['input']=sound
savemat('/home/bertrand/Data/MatlabProg/AuditoryFilters/noise.mat',data)

nbr_cf=50
cf=erbspace(100*Hz,1000*Hz, nbr_cf) 
#cf=atleast_1d(1000.)#log_space(500*Hz, 1000*Hz, nbr_cf)#atleast_1d(1000)#

interval=1#len(sound)/2

##### Control Path ####

# wide band pass control
#Initialisation of the filter coefficients of the bandpass filter
x_cf=11.9*log10(0.8+cf/456)
f_shift=(10**((x_cf+1.2)/11.9)-0.8)*456-cf
filt_b=zeros((len(cf), 3, 3))
filt_a=zeros((len(cf), 3, 3))
poles=zeros((len(cf),3),dtype='complex')
wbw=cf/4
poles[:,0:3]=tile(-2*pi*wbw+1j*2*pi*(cf+f_shift),[3,1]).T  
poles=(1+poles/(2*samplerate))/(1-poles/(2*samplerate))  #bilinear transform
bfp=2*pi*cf
zz=exp(1j*bfp/samplerate)
gain_norm=abs((zz**2+2*zz+1)/(zz**2-zz*(poles[:,0]+conj(poles[:,0]))+poles[:,0]*conj(poles[:,0])))**3*2

filt_b[:,:,0]=vstack([ones(len(cf)),2*ones(len(cf)),ones(len(cf))]).T
filt_b[:,:,1]=vstack([ones(len(cf)),2*ones(len(cf)),ones(len(cf))]).T
filt_b[:,:,2]=vstack([ones(len(cf)),zeros(len(cf)),-ones(len(cf))]).T

for iorder in xrange(3):
    filt_a[:,:,iorder]=vstack([ones(len(cf)),real(-(squeeze(poles[:,iorder])+conj(squeeze(poles[:,iorder])))),real(squeeze(poles[:,iorder])*conj(squeeze(poles[:,iorder])))]).T
filt_b[:,:,2]=filt_b[:,:,2]/tile(gain_norm,[3,1]).T

BP_control=  LinearFilterbank(sound,filt_b,filt_a) #bandpass filter instantiation (a controller will vary its coefficients)


# first non linearity of control path
Acp,Bcp,Ccp=100.,2.5,0.60 
func_NL1_control=lambda x:sign(x)*Bcp*log(1.+Acp*abs(x)**Ccp)
NL1_control=FunctionFilterbank(BP_control,func_NL1_control)

# second non linearity of control path
asym,s0,x1,s1=7.,8.,5.,3. 
shift = 1./(1.+asym)
x0 = s0*log((1.0/shift-1)/(1+exp(x1/s1)))
gain80=10**(log10(cf)*0.5732 + 1.5220)
rgain=  10**( log10(cf)*0.4 + 1.9)
average_control=0.3357
nlgain= (gain80 - rgain)/average_control
func_NL2_control=lambda x:(1.0/(1.0+exp(-(x-x0)/s0)*(1.0+exp(-(x-x1)/s1)))-shift)*nlgain
NL2_control=FunctionFilterbank(NL1_control,func_NL2_control)

#control low pass filter (its output will be used to control the signal path)
fc_LP_control=800*Hz
LP_control=ButterworthFilterbank(NL2_control, nbr_cf, 3, fc_LP_control, btype='low')


#low pass filter for feedback to control band pass (its output will be used to control the control path)
fc_LP_fb=500*Hz
LP_feed_back= ButterworthFilterbank(LP_control, nbr_cf, 3,fc_LP_fb, btype='low')



#### Signal Path ####

#Initisialisation of the filter coefficients of the bandpass filter
fp1=1.0854*cf-106.0034
x_cf=11.9*log10(0.8+cf/456)
N=len(cf)
rgain=10**(log10(cf)*0.4 + 1.9)
fp1=1.0854*cf-106.0034
ta=10**(log10(cf)*1.0230 + 0.1607)
tb=10**(log10(cf)*1.4292 - 1.1550) - 1000
f_shift=(10**((x_cf+1.2)/11.9)-0.8)*456-cf
filt_b=zeros((len(fp1), 3, 10))
filt_a=zeros((len(fp1), 3, 10))
poles=zeros((len(fp1),10),dtype='complex')
a=-rgain+1j*fp1*2*pi
poles[:,0:4]=tile(-rgain+1j*fp1*2*pi,[4,1]).T
poles[:,4:8]=tile(real(poles[:,0])- ta+1j*(imag(poles[:,0])- tb),[4,1]).T
poles[:,8:10]=tile((real(poles[:,0])+real(poles[:,4]))*.5+1j*(imag(poles[:,0])+imag(poles[:,4]))*.5,[2,1]).T
zeroa=array(-10**( log10(cf)*1.5-0.9 ))

poles=(1+poles/(2*samplerate))/(1-poles/(2*samplerate))
zeroa=(1+zeroa/(2*samplerate))/(1-zeroa/(2*samplerate))

bfp=2*pi*cf
gain_norm=1
zz=exp(1j*bfp/samplerate)
for ii in xrange(10):
    gain_norm=gain_norm*abs((zz**2-zz*(-1+zeroa)-zeroa)/(zz**2-zz*(poles[:,ii]+conj(poles[:,ii]))+poles[:,ii]*conj(poles[:,ii])))

for iorder in xrange(10):
    filt_b[:,:,iorder]=vstack([ones(len(cf)),-(-1+zeroa),-zeroa]).T
    filt_a[:,:,iorder]=vstack([ones(N),real(-(squeeze(poles[:,iorder])+conj(squeeze(poles[:,iorder])))),real(squeeze(poles[:,iorder])*conj(squeeze(poles[:,iorder])))]).T

filt_b[:,:,9]=filt_b[:,:,9]/tile(gain_norm,[3,1]).T


signal_path= LinearFilterbank(sound,filt_b,filt_a) #bandpass filter instantiation (a controller will vary its coefficients)


#### controlers #####

#definition of the class updater for the control path bandpass filter
class BP_control_update: 
    def __init__(self, target,samplerate,cf):
        self.target=target
        self.samplerate=samplerate
        self.cf=atleast_1d(cf)
        self.N=len(self.cf)
        self.iteration=0
        self.wbw=cf/4.
        self.p_realpart=-2*pi*self.wbw
        x_cf=11.9*log10(0.8+cf/456)
        self.K=10

        bfp=2*pi*cf
        self.zz=exp(1j*bfp/samplerate)
        self.num_gain=abs((zz**2+2*zz+1))
        self.zz2=self.zz**2              
        self.f_shift=(10**((x_cf+1.2)/11.9)-0.8)*456-cf

    def __call__(self,input):

         control_signal=input[-1,:]
         self.wbw=-(self.p_realpart-control_signal*self.K)/2.0/pi
         
         self.poles=tile(-2*pi*self.wbw+1j*2*pi*(self.cf+self.f_shift),[3,1]).T 
         self.poles=(1+self.poles/(2*samplerate))/(1-self.poles/(2*samplerate))

         for iorder in xrange(3):
             self.target.filt_a[:,:,iorder]=vstack([ones(self.N),real(-(squeeze(self.poles[:,iorder])+conj(squeeze(self.poles[:,iorder])))),real(squeeze(self.poles[:,iorder])*conj(squeeze(self.poles[:,iorder])))]).T
         gain_norm=self.num_gain/abs((self.zz2-self.zz*(poles[:,0]+conj(poles[:,0]))+poles[:,0]*conj(poles[:,0])))**3*2
         self.target.filt_b[:,:,2]=vstack([ones(len(cf)),zeros(len(cf)),-ones(len(cf))]).T/tile(gain_norm,[3,1]).T
         
#definition of the class updater for the signal path bandpass filter
class BP_signal_update: 
    def __init__(self, target,samplerate,cf):
        self.target=target
        self.samplerate=samplerate
        self.cf=atleast_1d(cf)
        self.N=len(self.cf)
        
        self.rgain=10**(log10(cf)*0.4 + 1.9)
        self.fp1=1.0854*cf-106.0034
        self.ta=10**(log10(cf)*1.0230 + 0.1607)
        self.tb=10**(log10(cf)*1.4292 - 1.1550) - 1000
        self.poles=zeros((len(cf),10),dtype='complex')
    def __call__(self,input):
        
         control_signal=input[-1,:]
         print -self.rgain-control_signal
         self.poles[:,0:4]=tile(-self.rgain-control_signal+1j*self.fp1*2*pi,[4,1]).T
         self.poles[:,4:8]=tile(real(self.poles[:,0])- self.ta+1j*(imag(self.poles[:,0])- self.tb),[4,1]).T
         self.poles[:,8:10]=tile((real(self.poles[:,0])+real(self.poles[:,4]))*.5+1j*(imag(self.poles[:,0])+imag(self.poles[:,4]))*.5,[2,1]).T
         self.poles=(1+self.poles/(2*self.samplerate))/(1-self.poles/(2*self.samplerate))
         for iorder in xrange(10):
            self.target.filt_a[:,:,iorder]=vstack([ones(self.N),real(-(squeeze(self.poles[:,iorder])+conj(squeeze(self.poles[:,iorder])))),real(squeeze(self.poles[:,iorder])*conj(squeeze(self.poles[:,iorder])))]).T



updater_control_path=BP_control_update(BP_control,samplerate,cf) #instantiation of the updater for the control path

#control1 = ControlFilterbank(NL1_control, LP_feed_back, BP_control,updater_control_path, interval)  #controler for the band pass filter of the control path
control1 = ControlFilterbank(signal_path, LP_feed_back, BP_control,updater_control_path, interval)  #controler for the band pass filter of the control path

updater_signal_path=BP_signal_update(signal_path,samplerate,cf)  #instantiation of the updater for the signal path
control2 = ControlFilterbank(control1,  LP_control, signal_path,updater_signal_path, interval)     #controler for the band pass filter of the signal path

#NL2_control.buffer_init()
#signal=NL2_control.buffer_fetch(0, len(sound))

#BP_control.buffer_init()
#signal=BP_control.buffer_fetch(0, len(sound))


signal=control1.buffer_fetch(0, len(sound))
t1=time()
#signal=signal_path.buffer_fetch(0, len(sound))
print 'the simulation took',time()-t1,' seconds to run'
#signal=vary_filter_coeff.level_dB[:,:len(sound)].T
#print signal.shape
data=dict()
data['out']=signal.T
savemat('/home/bertrand/Data/MatlabProg/AuditoryFilters/pmfr_BH.mat',data)

figure()
#plot(signal)
imshow(flipud(signal.T),aspect='auto')    
show()