'''
Implementation example of the compressive gammachirp auditory filter as described in  Irino, T. and Patterson R.,
"A compressive gammachirp auditory filter for both physiological and psychophysical data", JASA 2001.

A class called DCGC implementing this model is available in the library.

Technical implementation details and notation can be found in Irino, T. and Patterson R., "A Dynamic Compressive Gammachirp Auditory Filterbank",
IEEE Trans Audio Speech Lang Processing.
'''

from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *


simulation_duration=50*ms
samplerate=50*kHz
dBlevel=50*dB  # dB level of the input sound in rms dB SPL
sound=whitenoise(simulation_duration,samplerate=44*kHz).ramp() #generation of a white noise
sound=sound.atlevel(dBlevel) #set the sound to a certain dB level

nbr_cf=50  #number of centre frequencies
cf=erbspace(100*Hz,1000*Hz, nbr_cf)  #center frequencies with a spacing following an ERB scale


c1=-2.96 #glide slope of the first filterbank
b1=1.81  #factor determining the time constant of the first filterbank
c2=2.2   #glide slope of the second filterbank
b2=2.17  #factor determining the time constant of the second filterbank

order_ERB=4
ERBrate= 21.4*log10(4.37*cf/1000+1)
ERBwidth= 24.7*(4.37*cf/1000 + 1)
ERBspace = mean(diff(ERBrate))

update_interval=1  # the filter coefficients are updated every update_interval (here in sample)

param=dict() #definition of the parameters used in the control path output levels computation (see IEE paper for details)
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

#bank of passive gammachirp filters. As the control path uses the same passive filterbank than the signal path (but shifted in frequency)
#this filterbank is used by both pathway.
pGc=LogGammachirp(sound,cf,b=b1, c=c1)

fp1 = cf + c1*ERBwidth*b1/order_ERB #centre frequency of the signal path

#### Control Path ####

#the first filterbank in the control path consists of gammachirp filters
lct_ERB=1.5  #value of the shift in ERB frequencies of the control path with respect to the signal path
n_ch_shift  = round(lct_ERB/ERBspace); #value of the shift in channels
indch1_control = minimum(maximum(1, arange(1,nbr_cf+1)+n_ch_shift),nbr_cf).astype(int)-1 #index of the channel of the control path taken from pGc
fp1_control = fp1[indch1_control]
pGc_control=RestructureFilterbank(pGc,indexmapping=indch1_control) #the control path bank pass filter uses the channels of pGc indexed by indch1_control

#the second filterbank in the control path consists of fixed asymmetric compensation filters
frat_control=1.08
fr2_control = frat_control*fp1_control
asym_comp_control=Asymmetric_Compensation(pGc_control, fr2_control,b=b2, c=c2)
  

#definition of the controller class. What is does it take the outputs of the first and second fitlerbanks of the control filter as input, compute an overall intensity level 
#for each frequency channel. It then uses those level to update the filter coefficient of its target, the asymmetric compensation filterbank of the signal path.
class CompensensationFilterUpdater: 
    def __init__(self, target,fs,fp1,param):
        fp1=atleast_1d(fp1)
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
        #definition of the pole of the asymmetric comensation filters
        self.p0=2
        self.p1=1.7818*(1-0.0791*self.b)*(1-0.1655*abs(self.c))
        self.p2=0.5689*(1-0.1620*self.b)*(1-0.0857*abs(self.c))
        self.p3=0.2523*(1-0.0244*self.b)*(1+0.0574*abs(self.c))
        self.p4=1.0724
        
    def __call__(self,*input):
         value1=input[0][-1,:]
         value2=input[1][-1,:]
         level1 = maximum(maximum(value1,0),self.level1_prev*self.exp_deca_val) #the current level value is chosen as the max between the current output and the previous one decreased by a decay 
         level2 = maximum(maximum(value2,0),self.level2_prev*self.exp_deca_val)

         self.level1_prev=level1 #the value is stored for the next iteration
         self.level2_prev=level2
         #the overall intensity is computed between the two filterbank outputs
         level_total=self.lev_weight*self.level_ref*(level1/self.level_ref)**self.level_pwr1+(1-self.lev_weight)*self.level_ref*(level2/self.level_ref)**self.level_pwr2
         level_dB=20*log10(maximum(level_total,self.level_min))+self.RMStoSPL #then it is converted in dB           
         frat = self.frat0 + self.frat1*level_dB          #the frequency factor is calculated       
         fr2 = self.fp1*frat  #the centre frequency of the asymmetric compensation filters are updated
         self.target.filt_b, self.target.filt_a=asymmetric_compensation_coefs(self.samplerate,fr2,self.target.filt_b,self.target.filt_a,self.b,self.c,self.p0,self.p1,self.p2,self.p3,self.p4)
                 

#### Signal Path ####
#the signal path consists of the passive gammachirp filterbank pGc previously defined 
#followed by a asymmetric compensation filterbank
fr1=fp1*param['frat0']
varyingfilter_signal_path= Asymmetric_Compensation(pGc, fr1,b=b2, c=c2)
updater = CompensensationFilterUpdater(varyingfilter_signal_path,samplerate,fp1,param)   #the updater class is instantiated
 #the controler which takes the two filterbanks of the control path as inputs and the varying filter of the signal path as target is instantiated
control = ControlFilterbank(varyingfilter_signal_path, [pGc_control,asym_comp_control], varyingfilter_signal_path, updater, update_interval)  

#run the simulation
signal=control.buffer_fetch(0, len(sound))  #processing. Remember that the controler are at the end of the chain and the output of the whole path comes from them



figure()
imshow(flipud(signal.T),aspect='auto')    
show()