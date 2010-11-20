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

simulation_duration=500*ms
samplerate=50*kHz
sound = whitenoise(simulation_duration,samplerate).ramp()
#simulation_duration=50*ms+0*ms
data=dict()
data['input']=sound
savemat('/home/bertrand/Data/MatlabProg/AuditoryFilters/noise.mat',data)

nbr_cf=50
cf=erbspace(100*Hz,1000*Hz, nbr_cf) 
cf=log_space(500*Hz, 1000*Hz, nbr_cf)#atleast_1d(1000)#
fp1=1.0854*cf-106.0034
interval_change=len(sound)/2
print interval_change
##### Control Path ####
#### wide band pass control
#BP_control
#
#### first non linearity of control path
#Acp,Bcp,Ccp=100.,2.5,0.60 
#func_NL1_control=lambda x:sign(x)*Bcp*log(1+Acp*abs(x)**Ccp)
#NL1_control=FunctionFilterbank(BP_control,func_NL1_control)
#
#### second non linearity of control path
#asym,s0,x1,s1=7.,8.,5.,3. 
#shift = 1./(1.+asym)
#x0 = s0*log((1.0/shift-1)/(1+exp(x1/s1)));
#func_NL1_control=lambda x:(1.0/(1.0+exp(-(x-x0)/s0)*(1.0+exp(-(x-x1)/s1)))-shift)*nlgain
#NL2_control=NL1_control=FunctionFilterbank(NL1_control,func_NL2_control)
#
#### control low pass filter
fc_LP_control=800*Hz
LP_control= LowPassFilterbank(sound, nbr_cf,fc_LP_control)
#
#### low pass filter for feedback to control band pass
#fc_LP_fb=500*Hz
#LP_fb= LowPassFilterbank(NL2_control, nbr_cf,fc_LP_fb)
gain80=10**(log10(cf)*0.5732 + 1.5220)
average_control=0.3357
param=dict()
param['zero_r']=10**( log10(cf)*1.5-0.9 ) 


param['delay']=25
param['gain']=10**(log10(cf)*0.4 + 1.9)
param['nlgain']=(gain80 - param['gain'])/average_control
param['fp1']=fp1
param['ta']=10**(log10(cf)*1.0230 + 0.1607)
param['tb']=10**(log10(cf)*1.4292 - 1.1550) - 1000

def update_poles(rgain,poles,fp1,ta,tb):
    

    poles[:,0]=-rgain+1j*fp1*2*pi
    poles[:,0]=-rgain-1j*fp1*2*pi
    poles[:,4]=real(poles[:,0])- ta+1j*(imag(poles[:,0])- tb)
    poles[:,2]=(real(poles[:,0])+real(poles[:,4]))*.5+1j*(imag(poles[:,0])+imag(poles[:,4]))*.5
    
    poles[:,1] = conj(poles[:,0])
    poles[:,3] = conj(poles[:,2])
    poles[:,5] = conj(poles[:,4])
    
    poles[:,6] = poles[:,0]
    poles[:,7] = poles[:,1]
    poles[:,8] = poles[:,4]
    poles[:,9]= poles[:,5]
##    for i in xrange(10):
##        print poles[:,i]
##    exit()
    #poles[:,10:]=poles[:,:10]
  
    return poles
    
class BP_signal_update: 
    def __init__(self, fs,cf,control,param):
        fs=50e3
        self.cf=atleast_1d(cf)
        self.iteration=0
        self.control=control
        self.control.buffer_init()
        
        self.order_of_pole=10
        half_order_pole=10
        self.order_of_zero=10
        self.rgain=param['gain']
        self.fp1=param['fp1']
        self.ta=param['ta']
        self.tb=param['tb']
        
        self.filt_b=zeros((len(fp1), 11, 1))
        self.filt_a=zeros((len(fp1), 11, 1))
        self.poles=zeros((len(fp1),self.order_of_pole),dtype='complex')
        self.poles=update_poles(self.rgain,self.poles,self.fp1,self.ta,self.tb)
        self.zeroa=tile(array(-param['zero_r']),(self.order_of_zero,1)).T

        self.bfp=2*pi*self.cf
        gain_norm=1
        for ii in xrange(self.order_of_pole):
            gain_norm=gain_norm*((self.bfp-imag(self.poles[:,ii]))**2+real(self.poles[:,ii])**2)

        self.gain_norm=sqrt(gain_norm)/(sqrt(self.bfp**2+self.zeroa[:,0]**2))**self.order_of_zero;
        t1=time()         

#        
        self.poles=(1+self.poles/(2*fs))/(1-self.poles/(2*fs))
        self.zeroa=(1+self.zeroa/(2*fs))/(1-self.zeroa/(2*fs))

        #self.poles=self.poles[:,:12]
#        print self.poles
#        exit()
        #self.zeroa=self.zeroa[:,::2]
        
        
        for ich in xrange(len(fp1)):
            b,a=zpk2tf(self.zeroa[ich,:], self.poles[ich,:],1)
            print b.shape,a.shape
            self.filt_b[ich,:,0], self.filt_a[ich,:,0]=zpk2tf(self.zeroa[ich,:], self.poles[ich,:],1)# self.gain_norm[ich])
            
#        self.filt_a[0,0,0]=0
#        self.filt_a[0,1,0]=1
#        self.filt_b[0,0,0]=2
#        self.filt_b[0,1,0]=10
#        print self.filt_a
   
        print time()-t1

    def __call__(self):
        
         self.buffer_start += self.sub_buffer_length
         control_signal=zeros((1,len(fp1)))#self.control.buffer_fetch(self.buffer_start, self.buffer_start+self.sub_buffer_length)
         
#         self.poles=update_poles(-self.rgain-control_signal[-1,:],self.poles,self.fp1,self.ta,self.tb)
         t1=time()         
 
#             self.filt_b[ich,:,0], self.filt_a[ich,:,0]=bilinear(self.zeroa[ich], self.poles[ich,:])

#             print self.zeroa[ich]
#             self.filt_b[ich,0:2,0], self.filt_a[ich,:,0]=zpk2tf(self.zeroa[ich], self.poles[ich,:], self.gain_norm[ich])
#             self.filt_b[ich,0:2,0], self.filt_a[ich,:,0]=bilinear(self.filt_b[ich,0:2,0], self.filt_a[ich,:,0])
         print time()-t1

#### Signal Path ####
vary_coeff_BP_signal=BP_signal_update(samplerate,fp1,LP_control,param)
BP_signal= TimeVaryingIIRFilterbank(sound,interval_change,vary_coeff_BP_signal)


BP_signal.buffer_init()
signal=BP_signal.buffer_fetch(0, len(sound))

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