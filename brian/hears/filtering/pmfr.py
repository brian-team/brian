from brian import *
from filterbank import Filterbank,FunctionFilterbank,ControlFilterbank, CombinedFilterbank
from filterbanklibrary import *
from linearfilterbank import *
import warnings

try:
    from scikits.samplerate import resample
    have_scikits_samplerate = True
except (ImportError, ValueError):
    have_scikits_samplerate = False
#print have_scikits_samplerate

def set_parameters(cf,param):
    
    parameters=dict()
    parameters['fc_LP_control']=800*Hz
    parameters['fc_LP_fb']=500*Hz
    parameters['fp1']=1.0854*cf-106.0034
    parameters['ta']=10**(log10(cf)*1.0230 + 0.1607)
    parameters['tb']=10**(log10(cf)*1.4292 - 1.1550) - 1000
    parameters['gain80']=10**(log10(cf)*0.5732 + 1.5220)
    parameters['rgain']=10**( log10(cf)*0.4 + 1.9)
    parameters['average_control']=0.3357
    parameters['zero_r']= array(-10**( log10(cf)*1.5-0.9 ))   
        
    if param: 
        if not isinstance(param, dict): 
            raise Error('given parameters must be a dict')
        for key in param.keys():
            if not parameters.has_key(key):
                raise Exception(key + ' is invalid key entry for given parameters')
            parameters[key] = param[key]
    parameters['nlgain']= (parameters['gain80'] - parameters['rgain'])/parameters['average_control']
    return parameters

#### controlers #####

#definition of the class updater for the control path bandpass filter
class BP_control_update: 
    def __init__(self, target,samplerate,cf,param):
        parameters=set_parameters(cf,param)
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
        self.num_gain=abs((self.zz**2+2*self.zz+1))
        self.zz2=self.zz**2              
        self.f_shift=(10**((x_cf+1.2)/11.9)-0.8)*456-cf

    def __call__(self,input):

         control_signal=input[-1,:]
         self.wbw=-(self.p_realpart-control_signal*self.K)/2.0/pi
         
         self.poles=tile(-2*pi*self.wbw+1j*2*pi*(self.cf+self.f_shift),[3,1]).T 
         self.poles=(1+self.poles/(2*self.samplerate))/(1-self.poles/(2*self.samplerate))

         for iorder in xrange(3):
             self.target.filt_a[:,:,iorder]=vstack([ones(self.N),real(-(squeeze(self.poles[:,iorder])+conj(squeeze(self.poles[:,iorder])))),real(squeeze(self.poles[:,iorder])*conj(squeeze(self.poles[:,iorder])))]).T
         gain_norm=self.num_gain/abs((self.zz2-self.zz*(self.poles[:,0]+conj(self.poles[:,0]))+self.poles[:,0]*conj(self.poles[:,0])))**3*2
         self.target.filt_b[:,:,2]=vstack([ones(len(self.cf)),zeros(len(self.cf)),-ones(len(self.cf))]).T/tile(gain_norm,[3,1]).T
         
#definition of the class updater for the signal path bandpass filter
class BP_signal_update: 
    def __init__(self, target,samplerate,cf,param):
        parameters=set_parameters(cf,param)
        self.target=target
        self.samplerate=samplerate
        self.cf=atleast_1d(cf)
        self.N=len(self.cf)      
        self.rgain=parameters['rgain']
        self.fp1=parameters['fp1']
        self.ta=parameters['ta']
        self.tb=parameters['tb']
        self.poles=zeros((len(cf),10),dtype='complex')
    def __call__(self,input):        
         control_signal=input[-1,:]
         self.poles[:,0:4]=tile(-self.rgain-control_signal+1j*self.fp1*2*pi,[4,1]).T
         self.poles[:,4:8]=tile(real(self.poles[:,0])- self.ta+1j*(imag(self.poles[:,0])- self.tb),[4,1]).T
         self.poles[:,8:10]=tile((real(self.poles[:,0])+real(self.poles[:,4]))*.5+1j*(imag(self.poles[:,0])+imag(self.poles[:,4]))*.5,[2,1]).T
         self.poles=(1+self.poles/(2*self.samplerate))/(1-self.poles/(2*self.samplerate))
         for iorder in xrange(10):
            self.target.filt_a[:,:,iorder]=vstack([ones(self.N),real(-(squeeze(self.poles[:,iorder])+conj(squeeze(self.poles[:,iorder])))),real(squeeze(self.poles[:,iorder])*conj(squeeze(self.poles[:,iorder])))]).T

class PMFR(CombinedFilterbank):
    '''
    Class implementing the nonlinear auditory filterbank model as described in
    Tan, G. and Carney, L., 
    "A phenomenological model for the responses of auditory-nerve
    fibers. II. Nonlinear tuning with a frequency glide", JASA 2003.
    
    The model consists of a control path and a signal path. The control path
    controls both its own bandwidth via a feedback
    loop and also the bandwidth of the signal path. 
    
    Initialised with arguments:
    
    ``source``
        Source of the cochlear model.
        
    ``cf``
        List or array of center frequencies.
        
    ``update_interval``
        Interval in samples controlling how often the band pass filter of the
        signal pathway is updated. Smaller values are more accurate but
        increase the computation time.
        
    ``param``
        Dictionary used to overwrite the default parameters given in the
        original paper. 
    '''
    
    def __init__(self, source,cf,update_interval,param={}):
        
        CombinedFilterbank.__init__(self, source)
        source = self.get_modified_source()
        
        cf = atleast_1d(cf)
        nbr_cf=len(cf)
        parameters=set_parameters(cf,param)
        if int(source.samplerate)!=50000:
            warnings.warn('To use the PMFR cochlear model the sample rate should be 50kHz')
            if not have_scikits_samplerate:
                raise ImportError('To use the PMFR cochlear model the sample rate should be 50kHz and scikits.samplerate package is needed for resampling')               
            #source=source.resample(50*kHz)
            warnings.warn('The input to the PMFR cochlear model has been resampled to 50kHz')

        samplerate=source.samplerate
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
        BP_control=  LinearFilterbank(source,filt_b,filt_a) #bandpass filter instantiation (a controller will vary its coefficients)       
        
        # first non linearity of control path
        Acp,Bcp,Ccp=100.,2.5,0.60 
        func_NL1_control=lambda x:sign(x)*Bcp*log(1.+Acp*abs(x)**Ccp)
        NL1_control=FunctionFilterbank(BP_control,func_NL1_control)        
        
        # second non linearity of control path
        asym,s0,x1,s1=7.,8.,5.,3. 
        shift = 1./(1.+asym)
        x0 = s0*log((1.0/shift-1)/(1+exp(x1/s1)))
        func_NL2_control=lambda x:(1.0/(1.0+exp(-(x-x0)/s0)*(1.0+exp(-(x-x1)/s1)))-shift)*parameters['nlgain']
        NL2_control=FunctionFilterbank(NL1_control,func_NL2_control)      
        
        #control low pass filter (its output will be used to control the signal path)
        LP_control=Butterworth(NL2_control, nbr_cf, 3, parameters['fc_LP_control'], btype='low')       
        
        #low pass filter for feedback to control band pass (its output will be used to control the control path)
        LP_feed_back= Butterworth(LP_control, nbr_cf, 3,parameters['fc_LP_fb'], btype='low')
        
        
        #### Signal Path ####
        
        #Initisialisation of the filter coefficients of the bandpass filter
        x_cf=11.9*log10(0.8+cf/456)
        N=len(cf)
        f_shift=(10**((x_cf+1.2)/11.9)-0.8)*456-cf
        filt_b=zeros((len(cf), 3, 10))
        filt_a=zeros((len(cf), 3, 10))
        poles=zeros((len(cf),10),dtype='complex')
        a=-parameters['rgain']+1j*parameters['fp1']*2*pi
        poles[:,0:4]=tile(-parameters['rgain']+1j*parameters['fp1']*2*pi,[4,1]).T
        poles[:,4:8]=tile(real(poles[:,0])- parameters['ta']+1j*(imag(poles[:,0])- parameters['tb']),[4,1]).T
        poles[:,8:10]=tile((real(poles[:,0])+real(poles[:,4]))*.5+1j*(imag(poles[:,0])+imag(poles[:,4]))*.5,[2,1]).T
        zero_r=parameters['zero_r']
        poles=(1+poles/(2*samplerate))/(1-poles/(2*samplerate))
        zero_r=(1+zero_r/(2*samplerate))/(1-zero_r/(2*samplerate))   
        bfp=2*pi*cf
        gain_norm=1
        zz=exp(1j*bfp/samplerate)
        for ii in xrange(10):
            gain_norm=gain_norm*abs((zz**2-zz*(-1+zero_r)-zero_r)/(zz**2-zz*(poles[:,ii]+conj(poles[:,ii]))+poles[:,ii]*conj(poles[:,ii])))
        
        for iorder in xrange(10):
            filt_b[:,:,iorder]=vstack([ones(len(cf)),-(-1+zero_r),-zero_r]).T
            filt_a[:,:,iorder]=vstack([ones(N),real(-(squeeze(poles[:,iorder])+conj(squeeze(poles[:,iorder])))),real(squeeze(poles[:,iorder])*conj(squeeze(poles[:,iorder])))]).T 
        filt_b[:,:,9]=filt_b[:,:,9]/tile(gain_norm,[3,1]).T
        signal_path= LinearFilterbank(source,filt_b,filt_a) #bandpass filter instantiation (a controller will vary its coefficients)
        
        #controlers definition
        updater_control_path=BP_control_update(BP_control,samplerate,cf,param) #instantiation of the updater for the control path
        control1 = ControlFilterbank(signal_path, LP_feed_back, BP_control,updater_control_path, update_interval)  #controler for the band pass filter of the control path
        
        updater_signal_path=BP_signal_update(signal_path,samplerate,cf,param)  #instantiation of the updater for the signal path
        control2 = ControlFilterbank(control1,  LP_control, signal_path,updater_signal_path, update_interval)     #controler for the band pass filter of the signal path
        
        self.set_output(control2)