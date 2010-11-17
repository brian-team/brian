
from brian import *
from time import  time
#c06194a991
from scipy.io import savemat
from brian.hears import*
from brian.hears import filtering
filtering.use_gpu = False
#set_global_preferences(useweave=True)

samplerate=100*kHz
defaultclock.dt = 1/samplerate
simulation_duration=50*ms
dBlevel=60  # dB level in rms dB SPL

samplerate,sound=get_wav('/home/bertrand/Data/Toolboxes/AIM2006-1.40/Sounds/aimmat.wav')
sound=Sound(sound,samplerate)
sound=sound.setintensity(dBlevel)
#sound.atintensity(dBlevel)
print 'fs=',samplerate,'duration=',len(sound)/samplerate
simulation_duration=len(sound)/samplerate
defaultclock.dt = 1/samplerate


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

fp1 = cf + c1*ERBwidth*b1/order
print fp1
#print fp1,cf
#### Control Path ####

#bandpass filter (second order  gammatone filter)
pgammatone =GammatoneFilterbank(samplerate,cf,b=b1 )
pasym_comp=Asym_Comp_Filterbank(samplerate, cf, c=c1,asym_comp_order=4,b=b1)
pGc=FilterbankChain([pgammatone,pasym_comp])

pgammatone2 =GammatoneFilterbank(samplerate,cf,b=b1 )
pasym_comp2=Asym_Comp_Filterbank(samplerate, cf, c=c1,asym_comp_order=4,b=b1)
pGc2=FilterbankChain([pgammatone2,pasym_comp2])

level1_fb=FilterbankGroup(pGc2, sound)   
# control
lct_ERB=1.5
n_ch_shift  = round(lct_ERB/ERBspace);
indch_control = minimum(maximum(1, arange(1,nbr_cf+1)+n_ch_shift),nbr_cf).astype(int)-1

fp1_control = fp1[indch_control]

frat_control=1.08
fr2_control = frat_control*fp1_control

asym_comp_control3=Asym_Comp_Filterbank(samplerate, fr2_control, c=c2,asym_comp_order=4,b=b2)
control_path=FilterbankChain([pGc2,asym_comp_control3])
level2_fb=FilterbankGroup(control_path,sound)   

param=dict()
param['decay_tcst'] =.5*ms
param['b']=b2
param['c']=c2
param['order']=1
param['lev_weight']=.5
param['level_ref']=50
param['level_pwr1']=1.5
param['level_pwr2']=.5
param['RMStoSPL']=30
param['frat0']=.2330
param['frat1']=.005

#signal pathway
class AsymCompUpdate: 
    def __init__(self, fs,fp1,s1,s2,param):
        fp1=atleast_1d(fp1)
        
        self.s1=s1
        self.s2=s2
        self.fp1=fp1
        self.exp_deca_val = exp(-1/(param['decay_tcst'] *fs)*log(2))
        self.level_min = 10**(- param['RMStoSPL']/20)
        self.level_ref  = 10**(( param['level_ref'] - param['RMStoSPL'])/20) 
        self.b=param['b']
        self.c=param['c']
        self.order=param['order']
        self.lev_weight=param['lev_weight']
        self.level_ref=param['level_ref']
        self.level_pwr1=param['level_pwr1']
        self.level_pwr2=param['level_pwr2']
        self.level_pwr2=param['level_pwr2']
        self.RMStoSPL=param['RMStoSPL']
        self.frat0=param['frat0']
        self.frat1=param['frat1']
        
        self.level1_prev=-100
        self.level2_prev=-100
        
        self.filt_b=zeros((len(fp1), 2*self.order+1, 4))
        self.filt_a=zeros((len(fp1), 2*self.order+1, 4))
        self.p0=2
        self.p1=1.7818*(1-0.0791*self.b)*(1-0.1655*abs(self.c))
        self.p2=0.5689*(1-0.1620*self.b)*(1-0.0857*abs(self.c))
        self.p3=0.2523*(1-0.0244*self.b)*(1+0.0574*abs(self.c))
        self.p4=1.0724
        
    def __call__(self):
         level1 = maximum(maximum(self.s1,0),self.level1_prev*self.exp_deca_val)
         level2 = maximum(maximum(self.s2,0),self.level2_prev*self.exp_deca_val)
         self.level1_prev=level1
         self.level2_prev=level2
         level_total=self.lev_weight*self.level_ref*(level1/self.level_ref)**self.level_pwr1+\
         (1-self.lev_weight)*self.level_ref*(level2/self.level_ref)**self.level_pwr2
         level_dB=20*log10(maximum(level_total,self.level_min))+self.RMStoSPL
         
         frat = self.frat0 + self.frat1*level_dB
         #print level_dB,frat
         fr2 = self.fp1#*frat
         self.filt_b, self.filt_a=Asym_Comp_Coeff(samplerate,fr2,self.filt_b,self.filt_a,self.b,self.c,self.order,self.p0,self.p1,self.p2,self.p3,self.p4)
                 

#### Signal Path ####
asym_signal= TimeVaryingIIRFilterbank2(sound.rate,nbr_cf,AsymCompUpdate(samplerate,fp1,level1_fb.output,level2_fb.output,param))

#asym_signal= TimeVaryingIIRFilterbank2(sound.rate,nbr_cf,AsymCompUpdate(samplerate,fp1,level1_fb.output,level2_fb.output,param))
#asym_signal=Asym_Comp_Filterbank(samplerate, fp1, c=c2,asym_comp_order=4,b=b2)


#nonlinear pathway
signal_path_fb=FilterbankChain([pGc,asym_signal])

#
##signal_path= FilterbankGroup(pGc,sound)
signal_path= FilterbankGroup(signal_path_fb,sound)
#
##dnrl= FilterbankGroup(dnrl_filter, sound)
signal_monitor = StateMonitor(signal_path, 'output', record=True)

t1=time()
run(simulation_duration)
print 'the simulation took %f sec to run' %(time()-t1)

brian_hears=signal_monitor.getvalues()
data=dict()
data['out']=brian_hears
savemat('/home/bertrand/Data/MatlabProg/AuditoryFilters/cdgc_BH.mat',data)



figure()
#
imshow(flipud(signal_monitor.getvalues()),aspect='auto')
#for ifrequency in range((nbr_cf)):
#    subplot(nbr_cf,1,ifrequency+1)
#    plot(time_axis*1000,dnrl_monitor [ifrequency])
#    xlabel('(ms)')
    
show()


