from brian import *
from filterbank import Filterbank,FunctionFilterbank,ControlFilterbank, CombinedFilterbank
from filterbanklibrary import *
from linearfilterbank import *
import warnings
from scipy.io import loadmat,savemat
from brian.hears import *

__all__=['TanCarney']

try:
    from scikits.samplerate import resample
    have_scikits_samplerate = True
except (ImportError, ValueError):
    have_scikits_samplerate = False
#print have_scikits_samplerate

def set_parameters(cf,param):
    
    parameters=dict()
    parameters['fc_LP_control']=800 #Hz
    parameters['fc_LP_fb']=500 #Hz
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


class Control_Coefficients:
    
    def __init__(self,cf,samplerate):
        self.cf = cf
        self.PI2 = 2*3.14159265358979
        self.nch=len(cf)
        self.fs_bilinear = 2.0*samplerate#*ones(self.nch)
#        self.fs_bilinear =tile(self.fs_bilinear.reshape(self.nch,-1),3)
        self.x_cf=11.9*log10(0.8+cf/456);
        self.f_shift=(pow(10,((self.x_cf+1.2)/11.9))-0.8)*456-cf
        self.wbw=cf/4.0
        self.filt_a = zeros((len(cf),3,4), order='F')
#        self.filt_a[:,0,:] = 1
        self.filt_b = zeros((len(cf),3,4), order='F')
        self.control_signal = 0
        self.preal = zeros((self.nch,6))
        self.pimg = zeros((self.nch,6))
        self.preal,self.pimg = self.analog_poles()
        
    def return_coefficients(self,control_signal):

        self.wbw=-(self.preal[:,0] - control_signal)/self.PI2
        self.gain_norm_bp=((self.PI2**2*sqrt(self.wbw**2 + self.f_shift**2)*sqrt( (2*self.cf+self.f_shift)**2 + self.wbw**2 ))**3)/sqrt(self.PI2**2*self.cf**2)
        self.gain_norm_bp =tile(self.gain_norm_bp.reshape(self.nch,-1),3)

        iord = [1,3,5]
        temp=(self.fs_bilinear-self.preal[:,iord])**2 + self.pimg[:,iord]**2
        self.filt_a[:,0,0:3] = 1.
        self.filt_a[:,1,0:3]= -2*(self.fs_bilinear**2-self.preal[:,iord]**2-self.pimg[:,iord]**2)/temp            
        self.filt_a[:,2,0:3] = ((self.fs_bilinear+self.preal[:,iord])**2+self.pimg[:,iord]**2)/temp
        self.filt_b[:,0,0:3] = 1./temp
        self.filt_b[:,1,0:3] = 2./temp  
        self.filt_b[:,2,0:3] = 1./temp
        
        self.filt_a[:,0,3] = 1.
        self.filt_a[:,1,3]= 1.      
        self.filt_a[:,2,3] = 0.
        self.filt_b[:,0,3] = self.fs_bilinear
        self.filt_b[:,1,3] = -self.fs_bilinear
        self.filt_b[:,2,3] = 0
        
        self.filt_b[:,:,3] = self.gain_norm_bp*self.filt_b[:,:,3]  
        return self.filt_b,self.filt_a

    def analog_poles(self):
        self.preal[:,0] = self.PI2*self.wbw
        self.preal[:,1] = -self.PI2*self.wbw
        self.preal[:,2] = self.preal[:,0]
        self.preal[:,3] = self.preal[:,1]
        self.preal[:,4] = self.preal[:,0]
        self.preal[:,5] = self.preal[:,1]

        self.pimg[:,0] = self.PI2*(self.cf+self.f_shift)
        self.pimg[:,1] = -self.PI2*(self.cf+self.f_shift)
        self.pimg[:,2] = self.pimg[:,0]
        self.pimg[:,3] = self.pimg[:,1]
        self.pimg[:,4] = self.pimg[:,0]
        self.pimg[:,5] = self.pimg[:,1]
        
        return self.preal,self.pimg
    
    
class Signal_Coefficients:
    
    def __init__(self,cf,samplerate,parameters):
        self.cf = cf
        self.PI2 = 2*3.14159265358979
        self.nch=len(cf)
        self.fs_bilinear = 2.0*samplerate#*ones(self.nch)
        
        self.order_of_pole = 20             
        self.half_order_pole = self.order_of_pole/2
        self.order_of_zero  = self.half_order_pole

        self.filt_a = zeros((len(cf),3,10), order='F')
        self.filt_b = zeros((len(cf),3,10), order='F')
        
        self.preal = zeros((self.nch,20))
        self.pimg = zeros((self.nch,20))
        
        self.control_signal = 0
        
        self.rgain =parameters['rgain']# 10**(log10(cf)*0.4 + 1.9)
        self.fp1=parameters['fp1']#1.0854*cf-106.0034
        self.ta= parameters['ta']#    10**(log10(cf)*1.0230 + 0.1607)
        self.tb= parameters['tb']#    10**(log10(cf)*1.4292 - 1.1550) - 1000
        self.zeroa = parameters['zero_r']# -10**(log10(cf)*1.5-0.9 )  
        
        self.zeroamat = tile(self.zeroa.reshape(self.nch,-1),10)
        self.preal,self.pimg = self.analog_poles(0)

        
        self.cfmat = tile(self.cf.reshape(self.nch,-1),20)  
        self.gain_norm = sqrt(prod((2*pi*self.cfmat-self.pimg[:,0:20])**2+self.preal[:,0:20]**2,axis=1))

        self.gain_norm = self.gain_norm /(sqrt((2*pi*self.cf)**2+self.zeroa**2))**self.order_of_zero
        self.gain_norm  = tile(self.gain_norm.reshape(self.nch,-1),3)

        
    def return_coefficients(self,control_signal):
        self.preal,self.pimg = self.analog_poles(control_signal)
        
        iord = arange(2,22,2)-1
        temp=(self.fs_bilinear-self.preal[:,iord])**2 + self.pimg[:,iord]**2
    
        self.filt_a[:,0,:] = 1
        self.filt_a[:,1,:] = -2*(self.fs_bilinear**2-self.preal[:,iord]**2-self.pimg[:,iord]**2)/temp            
        self.filt_a[:,2,:] = ((self.fs_bilinear+self.preal[:,iord])**2+self.pimg[:,iord]**2)/temp
        
        self.filt_b[:,0,:] = (-self.zeroamat+self.fs_bilinear)/temp
        self.filt_b[:,1,:] = (-2*self.zeroamat)/temp
        self.filt_b[:,2,:] = (-self.zeroamat-self.fs_bilinear)/temp
        self.filt_b[:,:,9] = self.gain_norm/3.*self.filt_b[:,:,9] 

        return self.filt_b,self.filt_a

    def analog_poles(self,control_signal):
        self.preal[:,0] = -self.rgain-control_signal
        self.preal[:,4] = self.preal[:,0]-self.ta
        self.preal[:,2] = (self.preal[:,0]+self.preal[:,4])*0.5
        self.preal[:,1] = self.preal[:,0]
        self.preal[:,3] = self.preal[:,2]
        self.preal[:,5] = self.preal[:,4]
        
        self.preal[:,6] = self.preal[:,0]
        self.preal[:,7] = self.preal[:,1]
        self.preal[:,8] = self.preal[:,4]
        self.preal[:,9] = self.preal[:,5]
        self.preal[:,10:] = self.preal[:,:10]
        
        self.pimg[:,0] = self.PI2*self.fp1
        self.pimg[:,4] = self.pimg[:,0]-self.tb
        self.pimg[:,2] = (self.pimg[:,0]+self.pimg[:,4])*0.5
        self.pimg[:,1] = -self.pimg[:,0]
        self.pimg[:,3] = -self.pimg[:,2]
        self.pimg[:,5] = -self.pimg[:,4]
        self.pimg[:,6] = self.pimg[:,0]
        self.pimg[:,7] = self.pimg[:,1]
        self.pimg[:,8] = self.pimg[:,4]
        self.pimg[:,9] = self.pimg[:,5]
        self.pimg[:,10:] = self.pimg[:,:10]

        return self.preal,self.pimg
    
class Filter_Update: 
    def __init__(self, target,coef):
        self.coef = coef
        self.target = target
        self.param = []
    def __call__(self,input):  
        self.target.filt_b,self.target.filt_a = self.coef.return_coefficients(input[-1,:]) 
        self.param.append(self.coef.control_signal)


class LowPass_IHC(LinearFilterbank):
    def __init__(self,source,cf,fc,gain,order): 
        nch = len(cf)
        TWOPI = 2*pi
        self.samplerate =  source.samplerate
        c = 2.0 * self.samplerate
        c1LP = ( c/Hz - TWOPI*fc ) / ( c/Hz + TWOPI*fc )
        c2LP = TWOPI*fc/Hz / (TWOPI*fc + c/Hz)
        
        b_temp = array([c2LP,c2LP])
        a_temp = array([1,-c1LP])
        
        filt_b = tile(b_temp.reshape([2,1]),[nch,1,order])               
        filt_a = tile(a_temp.reshape([2,1]),[nch,1,order]) 
        filt_b[:,:,0] = filt_b[:,:,0]*gain

        LinearFilterbank.__init__(self, source, filt_b, filt_a)

class LowPass_filter(LinearFilterbank):
    def __init__(self,source,cf,fc,gain,order):
        nch = len(cf)
        TWOPI = 2*pi
        self.samplerate =  source.samplerate
        c = 2.0 * self.samplerate
        c1LP = ( c/Hz - TWOPI*fc ) 
        
        b_temp = array([1,1])/ ( c/Hz + TWOPI*fc )
        a_temp = array([1,-c1LP/ ( c/Hz + TWOPI*fc )])
        
        filt_b = tile(b_temp.reshape([2,1]),[nch,1,order])               
        filt_a = tile(a_temp.reshape([2,1]),[nch,1,order]) 
        filt_b[:,:,order-1] = filt_b[:,:,order-1]*gain

        LinearFilterbank.__init__(self, source, filt_b, filt_a)

def saturation_fc(x,A0=1,B=1,C=1,D=1):
    ind = x>=0
    x[ind]=A0*log(x[ind]*B+1.0)
    ind = x<0
    dtemp = (-x[ind])**C
    tempA = -A0*(dtemp+D)/(3*dtemp+D);
    x[ind]=tempA*log(abs(x[ind])*B+1.0)

    return x
    

class TanCarney(CombinedFilterbank):
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
        samplerate=source.samplerate

        parameters=set_parameters(cf,param)
        if int(source.samplerate)!=50000:
            warnings.warn('To use the TanCarney cochlear model the sample rate should be 50kHz')
#            if not have_scikits_samplerate:
#                raise ImportError('To use the PMFR cochlear model the sample rate should be 50kHz and scikits.samplerate package is needed for resampling')               
#            #source=source.resample(50*kHz)
#            warnings.warn('The input to the PMFR cochlear model has been resampled to 50kHz'
          
#        ##### Control Path ####
        # band pass filter
        control_coef = Control_Coefficients(cf, samplerate)
        [filt_b,filt_a] = control_coef.return_coefficients(0)
        BP_control = LinearFilterbank(source,filt_b,filt_a)        
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
        gain_lp_con = (2*pi*parameters['fc_LP_control'])**3*1.5
        LP_control = LowPass_filter(NL2_control,cf,parameters['fc_LP_control'],gain_lp_con,3)   
        #low pass filter for feedback to control band pass (its output will be used to control the control path)
        gain_lp_fb = parameters['fc_LP_fb']*2*pi *10
        LP_feed_back = LowPass_filter(LP_control,cf,parameters['fc_LP_fb'],gain_lp_fb,1) 
         
         
        #### signal path  ####
        # band pass filter
        signal_coef = Signal_Coefficients(cf, samplerate,parameters)
        [filt_b,filt_a] = signal_coef.return_coefficients(0)
        BP_signal = LinearFilterbank(source,filt_b,filt_a)
        ## Saturation
        saturation = FunctionFilterbank(BP_signal,saturation_fc,A0=0.1,B=2000,C=1.74,D=6.87e-9)   
        ## low pass IHC     
        ihc = LowPass_IHC(saturation,cf,3800,1,7)
        
        ### controlers ###
        updater1=Filter_Update(BP_control,control_coef) #instantiation of the updater for the control path
        output1 = ControlFilterbank(ihc, LP_feed_back, BP_control,updater1, update_interval)  #controler for the band pass filter of the control path
        
        updater2=Filter_Update(BP_signal,signal_coef) #instantiation of the updater for the control path
        output2 = ControlFilterbank(output1, LP_control, BP_signal,updater2, update_interval)  #controler for the band pass filter of the control path
        
#
#        self.control_cont = updater1.param
#        self.signal_cont = updater2.param
#        self.set_output(BP_control)
        self.set_output(output2)
