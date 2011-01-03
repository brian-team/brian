from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *
from filterbank import Filterbank,FunctionFilterbank,ControlFilterbank,RestructureFilterbank
from filterbanklibrary import *
from linearfilterbank import *

def set_parameters(cf,given_param):
    
    parameters=dict()
    parameters['b1'] = 1.81
    parameters['c1'] = -2.96
    parameters['b2'] = 2.17
    parameters['c2'] = 2.2
    parameters['decay_tcst'] = .5*ms
    parameters['lev_weight'] = .5
    parameters['level_ref'] = 50.
    parameters['level_pwr1'] = 1.5
    parameters['level_pwr2'] = .5
    parameters['RMStoSPL'] = 30.
    parameters['frat0'] = .2330
    parameters['frat1'] = .005
    parameters['lct_ERB'] = 1.5  #value of the shift in ERB frequencies
    parameters['frat_control'] = 1.08
    parameters['order_gc']=4
    parameters['ERBrate']= 21.4*log10(4.37*cf/1000+1)
    parameters['ERBwidth']= 24.7*(4.37*cf/1000 + 1)
    
    if given_param: 
        if not isinstance(given_param, dict): 
            raise Error('given parameters must be a dict')
        for key in given_param.keys():
            if not parameters.has_key(key):
                raise Exception(key + ' is invalid key entry for given parameters')
            parameters[key] = given_param[key]

    return parameters

#defition of the controler class
class AsymCompUpdate: 
    def __init__(self,target,samplerate,fp1,param):
        fp1=atleast_1d(fp1)
        self.iteration=0
        self.target=target
        self.samplerate=samplerate
        self.fp1=fp1             
        self.exp_deca_val = exp(-1/(param['decay_tcst'] *samplerate)*log(2))
        self.level_min = 10**(- param['RMStoSPL']/20)
        self.level_ref  = 10**(( param['level_ref'] - param['RMStoSPL'])/20)        
        self.b=param['b2']
        self.c=param['c2']
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
 
class CDGC(Filterbank):
    '''
    Class implementing  the compressive gammachirp auditory filter as described in  Irino, T. and Patterson R.,
    "A compressive gammachirp auditory filter for both physiological and psychophysical data", JASA 2001
    
    Technical implementation details and notation can be found in Irino, T. and Patterson R., "A Dynamic Compressive Gammachirp Auditory Filterbank",
    IEEE Trans Audio Speech Lang Processing.
    
    The model consists of a control pathway with a bank of bandpass filters followed by a bank of highpass filters (this chain yields a bank of gammachirp filters).
    The signal pathway consist of a bank of fix bandpass filters followed by a bank of highpass filters with variable cutoff frequencies (this chain yields a bank gammachirp
    filters with a level-dependent bandwidth).The highpass filters of the signal pathway are controlled by the output levels of the two stages of the control pathway. 
    
    Initialised with arguments:
    
    ``source``
        Source of the cochlear model.
        
    ``cf``
        List or array of center frequencies.
        
    ``interval``
        interval in sample when the band pass filter of the signal pathway is updated
        
    ``given_param``
        dictionary used to overwrite the default parameters given in the original paper. . 
    
    '''
    
    def __new__(cls, source,cf,interval,given_param={}):
        
        parameters=set_parameters(cf,given_param)
        ERBspace = mean(diff(parameters['ERBrate']))
        cf=atleast_1d(cf)
        #bank of passive gammachirp filters. As the control path uses the same passive filterbank than the signal path (buth shifted in frequency)
        #this filterbanl is used by both pathway.
        pGc=LogGammachirpFilterbank(source,cf,b=parameters['b1'], c=parameters['c1'])
        fp1 = cf + parameters['c1']*parameters['ERBwidth']*parameters['b1']/parameters['order_gc']
        nbr_cf=len(cf)
        #### Control Path ####
        n_ch_shift  = round(parameters['lct_ERB']/ERBspace); #value of the shift in channels
        indch1_control = minimum(maximum(1, arange(1,nbr_cf+1)+n_ch_shift),nbr_cf).astype(int)-1
        fp1_control = fp1[indch1_control]        
        pGc_control=RestructureFilterbank(pGc,indexmapping=indch1_control)
        fr2_control = parameters['frat_control']*fp1_control       
        asym_comp_control=Asymmetric_Compensation_Filterbank(pGc_control, fr2_control,b=parameters['b2'], c=parameters['c2'])
        
        #### Signal Path ####
        fr1=fp1*parameters['frat0']
        signal_path= Asymmetric_Compensation_Filterbank(pGc, fr1,b=parameters['b2'], c=parameters['c2'])

        #### Controler #### 
        updater = AsymCompUpdate(signal_path,source.samplerate,fp1,parameters)   #the updater
        control = ControlFilterbank(signal_path, [pGc_control,asym_comp_control], signal_path, updater, interval)  
        
        return control