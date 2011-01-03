'''
Implementation example of the compressive gammachirp auditory filter as described in  Irino, T. and Patterson R.,
"A compressive gammachirp auditory filter for both physiological and psychophysical data", JASA 2001

Technical implementation details and notation can be found in Irino, T. and Patterson R., "A Dynamic Compressive Gammachirp Auditory Filterbank",
IEEE Trans Audio Speech Lang Processing.



'''

from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *
from scipy.io import savemat
from time import time


simulation_duration=50*ms
samplerate=50*kHz
sound = whitenoise(simulation_duration,samplerate)

print 'fs=',sound.samplerate,'duration=',len(sound)/sound.samplerate

simulation_duration=len(sound)/sound.samplerate 


#sound = whitenoise(simulation_duration,samplerate,dB=dBlevel).ramp()
nbr_cf=50
cf=erbspace(100*Hz,1000*Hz, nbr_cf) 
cf=log_space(100*Hz, 1000*Hz, nbr_cf)

order_ERB=4
c1=-2.96
b1=1.81
c2=2.2
b2=2.17

ERBrate= 21.4*log10(4.37*cf/1000+1)
ERBwidth= 24.7*(4.37*cf/1000 + 1)
ERBspace = mean(diff(ERBrate))
#print ERBspace

interval=1

param=dict()
param['decay_tcst'] =.5*ms
param['b2']=b2
param['c2']=c2
param['order']=1.
param['lev_weight']=.5
param['level_ref']=50.
param['level_pwr1']=1.5
param['level_pwr2']=.5
param['RMStoSPL']=30.
param['frat0']=.2330
param['frat1']=.005

#bank of passive gammachirp filters. As the control path uses the same passive filterbank than the signal path (buth shifted in frequency)
#this filterbanl is used by both pathway.
pGc=LogGammachirpFilterbank(sound,cf,b=b1, c=c1)

fp1 = cf + c1*ERBwidth*b1/order_ERB

#### Control Path ####

lct_ERB=1.5  #value of the shift in ERB frequencies
n_ch_shift  = round(lct_ERB/ERBspace); #value of the shift in channels
indch1_control = minimum(maximum(1, arange(1,nbr_cf+1)+n_ch_shift),nbr_cf).astype(int)-1
fp1_control = fp1[indch1_control]

#the control path bank pass filter uses the channels of pGc indexed by indch1_control
pGc_control=RestructureFilterbank(pGc,indexmapping=indch1_control)
frat_control=1.08
fr2_control = frat_control*fp1_control

asym_comp_control=Asymmetric_Compensation_Filterbank(pGc_control, fr2_control,b=b2, c=c2)
  



#defition of the controler class
class AsymCompUpdate: 
    def __init__(self, target,fs,fp1,param):
        fp1=atleast_1d(fp1)

        self.iteration=0
        self.target=target
        self.fp1=fp1             
        self.exp_deca_val = exp(-1/(param['decay_tcst'] *fs)*log(2))
        self.level_min = 10**(- param['RMStoSPL']/20)
        self.level_ref  = 10**(( param['level_ref'] - param['RMStoSPL'])/20) 
        self.samplerate=fs
        self.b=param['b2']
        self.c=param['c2']
        self.order=param['order']
        self.lev_weight=param['lev_weight']
        self.level_ref=param['level_ref']
        self.level_pwr1=param['level_pwr1']
        self.level_pwr2=param['level_pwr2']
        self.RMStoSPL=param['RMStoSPL']
        self.frat0=param['frat0']
        self.frat1=param['frat1']
        self.level1_prev=-100
        self.level2_prev=-100
        self.p0=2
        self.p1=1.7818*(1-0.0791*self.b)*(1-0.1655*abs(self.c))
        self.p2=0.5689*(1-0.1620*self.b)*(1-0.0857*abs(self.c))
        self.p3=0.2523*(1-0.0244*self.b)*(1+0.0574*abs(self.c))
        self.p4=1.0724
    def __call__(self,*input):
         value1=input[0][-1,:]
         value2=input[1][-1,:]
         level1 = maximum(maximum(value1,0),self.level1_prev*self.exp_deca_val)
         level2 = maximum(maximum(value2,0),self.level2_prev*self.exp_deca_val)

         self.level1_prev=level1
         self.level2_prev=level2
         level_total=self.lev_weight*self.level_ref*(level1/self.level_ref)**self.level_pwr1+(1-self.lev_weight)*self.level_ref*(level2/self.level_ref)**self.level_pwr2
         level_dB=20*log10(maximum(level_total,self.level_min))+self.RMStoSPL
                                        
         frat = self.frat0 + self.frat1*level_dB
         fr2 = self.fp1*frat
   
         self.iteration+=1
         self.target.filt_b, self.target.filt_a=asymmetric_compensation_coefs(self.samplerate,fr2,self.target.filt_b,self.target.filt_a,self.b,self.c,self.p0,self.p1,self.p2,self.p3,self.p4)
                 

#### Signal Path ####

fr1=fp1*param['frat0']
signal_path= Asymmetric_Compensation_Filterbank(pGc, fr1,b=b2, c=c2)

updater = AsymCompUpdate(signal_path,samplerate,fp1,param)   #the updater
control = ControlFilterbank(signal_path, [pGc_control,asym_comp_control], signal_path, updater, interval)  

#run the simulation
t1=time()
signal=control.buffer_fetch(0, len(sound))
print 'the simulation took',time()-t1,' seconds to run'


figure()
imshow(flipud(signal.T),aspect='auto')    
show()