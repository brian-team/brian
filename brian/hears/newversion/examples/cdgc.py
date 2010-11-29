from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *
from scipy.io import savemat
from time import time

dBlevel=10  # dB level in rms dB SPL
sound=Sound.load('/home/bertrand/Data/Toolboxes/AIM2006-1.40/Sounds/aimmat.wav')
samplerate=sound.samplerate
sound=sound.atintensity(dBlevel)
sound.samplerate=samplerate

data=dict()
data['input']=sound
savemat('/home/bertrand/Data/MatlabProg/AuditoryFilters/stimulus.mat',data)

print 'fs=',sound.samplerate,'duration=',len(sound)/sound.samplerate

simulation_duration=len(sound)/sound.samplerate 


#sound = whitenoise(simulation_duration,samplerate,dB=dBlevel).ramp()
nbr_cf=50
cf=erbspace(100*Hz,1000*Hz, nbr_cf) 
cf=log_space(100*Hz, 1000*Hz, nbr_cf)

order=4
c1=-2.96
b1=1.81
c2=2.2
b2=2.17

ERBrate= 21.4*log10(4.37*cf/1000+1)
ERBwidth= 24.7*(4.37*cf/1000 + 1)
ERBspace = mean(diff(ERBrate))
#print ERBspace

interval_change=1

fp1 = cf + c1*ERBwidth*b1/order

#print fp1,cf
#### Control Path ####

#bandpass filter (second order  gammatone filter)
pgammatone =GammatoneFilterbank(sound,cf,b=b1 )
pGc=Asym_Comp_Filterbank(pgammatone, cf, c=c1,asym_comp_order=4,b=b1)
#pGc=GammachirpIIRFilterbank(sound,cf, c=c1,asym_comp_order=4,b=b1)
   
# control
lct_ERB=1.5
n_ch_shift  = round(lct_ERB/ERBspace);
indch1_control = minimum(maximum(1, arange(1,nbr_cf+1)+n_ch_shift),nbr_cf).astype(int)-1
fp1_control = fp1[indch1_control]

pGc_control=RestructureFilterbank(pGc,indexmapping=indch1_control)
frat_control=1.08
fr2_control = frat_control*fp1_control

asym_comp_control=Asym_Comp_Filterbank(pGc_control, fr2_control, c=c2,asym_comp_order=4,b=b2)
  

param=dict()
param['indch1_control'] =indch1_control
param['decay_tcst'] =.5*ms
param['b']=b2
param['c']=c2
param['order']=1.
param['lev_weight']=.5
param['level_ref']=50.
param['level_pwr1']=1.5
param['level_pwr2']=.5
param['RMStoSPL']=30.
param['frat0']=.2330
param['frat1']=.005

#signal pathway
class AsymCompUpdate: 
    def __init__(self, fs,fp1,s1,s2,param):
        fp1=atleast_1d(fp1)

        self.iteration=0
        self.s1=s1
        self.s2=s2
        self.s1.buffer_init()
        self.s2.buffer_init()
        self.fp1=fp1             
        self.exp_deca_val = exp(-1/(param['decay_tcst'] *fs)*log(2))
        self.level_min = 10**(- param['RMStoSPL']/20)
        self.level_ref  = 10**(( param['level_ref'] - param['RMStoSPL'])/20) 
        
#        self.indch1_control=param['indch1_control'] 
        self.b=param['b']
        self.c=param['c']
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
        
        self.filt_b=zeros((len(fp1), 2*self.order+1, 4))
        self.filt_a=zeros((len(fp1), 2*self.order+1, 4))
        #self.level_dB=zeros((self.filt_b.shape[0],100000))
        self.p0=2
        self.p1=1.7818*(1-0.0791*self.b)*(1-0.1655*abs(self.c))
        self.p2=0.5689*(1-0.1620*self.b)*(1-0.0857*abs(self.c))
        self.p3=0.2523*(1-0.0244*self.b)*(1+0.0574*abs(self.c))
        self.p4=1.0724
#        self.filt_b, self.filt_a=Asym_Comp_Coeff(samplerate,self.fp1,self.filt_b,self.filt_a,self.b,self.c,self.order,self.p0,self.p1,self.p2,self.p3,self.p4)

    def __call__(self):
         self.buffer_start += self.sub_buffer_length
         #print self.buffer_start,self.buffer_start+self.sub_buffer_length
         value1=self.s1.buffer_fetch(self.buffer_start, self.buffer_start+self.sub_buffer_length)
         value2=self.s2.buffer_fetch(self.buffer_start, self.buffer_start+self.sub_buffer_length)#array([[1,2],[1,2]])#

         level1 = maximum(maximum(value1[-1,:],0),self.level1_prev*self.exp_deca_val)
         level2 = maximum(maximum(value2[-1,:],0),self.level2_prev*self.exp_deca_val)

         self.level1_prev=level1
         self.level2_prev=level2
         level_total=self.lev_weight*self.level_ref*(level1/self.level_ref)**self.level_pwr1+(1-self.lev_weight)*self.level_ref*(level2/self.level_ref)**self.level_pwr2
         level_dB=20*log10(maximum(level_total,self.level_min))+self.RMStoSPL
                                        
         
#         self.level_dB=level_dB
         frat = self.frat0 + self.frat1*level_dB
         #print level_dB,frat
         fr2 = self.fp1*frat
   
         #self.level_dB[:,self.iteration]=frat
         self.iteration+=1
         self.filt_b, self.filt_a=Asym_Comp_Coeff(samplerate,fr2,self.filt_b,self.filt_a,self.b,self.c,self.order,self.p0,self.p1,self.p2,self.p3,self.p4)
                 

#### Signal Path ####
vary_filter_coeff=AsymCompUpdate(samplerate,fp1,pGc_control,asym_comp_control,param)
signal_path= TimeVaryingIIRFilterbank(pGc,interval_change,vary_filter_coeff)

#pGc_control.buffer_init()
#signal=pGc_control.buffer_fetch(0, len(sound))
#pGc.buffer_init()
#signal=pGc.buffer_fetch(0, len(sound))

signal_path.buffer_init()
t1=time()
signal=signal_path.buffer_fetch(0, len(sound))
print 'the simulation took',time()-t1,' seconds to run'
#signal=vary_filter_coeff.level_dB[:,:len(sound)].T
#print signal.shape
data=dict()
data['out']=signal.T
savemat('/home/bertrand/Data/MatlabProg/AuditoryFilters/cdgc_BH.mat',data)

figure()
imshow(flipud(signal.T),aspect='auto')    
show()