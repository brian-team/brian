from brian import *
from filterbank import Filterbank,FunctionFilterbank,ControlFilterbank, CombinedFilterbank
from filterbanklibrary import *
from linearfilterbank import *
import warnings
from scipy.io import loadmat,savemat
from brian.hears import *


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

##
def gain_groupdelay(binwidth,centerfreq,cf,tau):
    
    tmpcos = cos(2*pi*(centerfreq-cf)*binwidth)
    dtmp2 = tau*2.0/binwidth
    c1LP = (dtmp2-1)/(dtmp2+1)
    c2LP = 1.0/(dtmp2+1)
    tmp1 = 1+c1LP*c1LP-2*c1LP*tmpcos
    tmp2 = 2*c2LP*c2LP*(1+tmpcos)
    
    wb_gain = sqrt(tmp1/tmp2)
    
    grdelay = floor((0.5-(c1LP*c1LP-c1LP*tmpcos)/(1+c1LP*c1LP-2*c1LP*tmpcos))).astype(int)
    
    return wb_gain,grdelay

def get_taubm(cf,CAgain,taumax):
    bwfactor = 0.7;
    factor   = 2.5;
    ratio  = 10**(-CAgain/(20.0*factor))
    bmTaumax = taumax/bwfactor;
    bmTaumin = bmTaumax*ratio;   
    return bmTaumax,bmTaumin,ratio

def get_tauwb(cf,CAgain,order):
    ####
    ratio = 10**(-CAgain/(20.0*order))     #ratio of TauMin/TauMax according to the gain, order */
    ##Q10 = pow(10,0.4708*log10(cf/1e3)+0.5469); */ /* 75th percentile */
    Q10 = 10**(0.4708*log10(cf/1e3)+0.4664) #/* 50th percentile */
    ##Q10 = pow(10,0.4708*log10(cf/1e3)+0.3934); */ /* 25th percentile */
    bw     = cf/Q10;
    taumax = 2.0/(2*pi*bw);
    taumin   = taumax*ratio;
    return taumax,taumin


### function to initialize the chirp filters
class Control_Coefficients:
    
    def __init__(self,cf,samplerate):
        self.nch=len(cf)
        self.fs_bilinear = 2.0*samplerate;
        self.x_cf=11.9*log10(0.8+cf/456);
        self.f_shift=(pow(10,((x_cf+1.2)/11.9))-0.8)*456-cf
        self.wbw=cf/4.0
        self.filt_a = zeros((len(cf),3,4), order='F')
        self.filt_a[:,0,:] = 1
        self.filt_b = zeros((len(cf),3,4), order='F')
        
    def return_coefficients(self):
        
        self.preal,self.pimg = self.analog_poles()
        self.wbw=-(self.preal[:,0] - control_nl)/self.PI2
        self.gain_norm_bp=(self.PI2**2*sqrt(self.wbw**2 + self.f_shift**2)*sqrt( (2*self.cf+self.f_shift)**2 + wbw**2 ))**3
#        gain_norm_bp=gain_norm_bp/
#                 pow( sqrt(2*PI*z[1]*2*PI*z[1] + 2*PI*cf*2*PI*cf), z_order );
        iord = [0,2,1,0,1]
        temp=(fs_bilinear-preal)**2 + pimg**2;    
        self.filt_a[:,0,0:2] = 1
        self.filt_a[:,1,0:2]= -2*(self.fs_bilinear**2-self.preal[:,iord]**2-self.pimg[:,iord]**2)/temp            
        self.filt_a[:,2,0:2] = ((self.fs_bilinear+self.preal[:,iord])**2+self.pimg[:,iord]**2)/temp
        self.filt_b[:,0,0:2] = 1/temp
        self.filt_b[:,1,0:2] = 2./temp  
        self.filt_b[:,2,0:2] = 1./temp
        
        self.filt_a[:,0,3] = 1
        self.filt_a[:,1,3]= 0           
        self.filt_a[:,2,3] = 0
        self.filt_b[:,0,3] = 1
        self.filt_b[:,1,3] = 0
        self.filt_b[:,2,3] = 0
        
        self.filt_b[:,:,4] = self.gain_norm_bp*self.filt_b[:,:,4]           
                 
        return self.filt_b,self.filt_a

    def analog_poles(self):
        self.preal[:,0] = self.PI2*self.wbw  #0
        self.preal[:,1] = -self.PI2*self.wbw
        self.preal[:,2] = self.preal[:,0]
        self.preal[:,3] = self.preal[:,1]
        self.preal[:,4] = sself.preal[:,0]
        self.preal[:,5] = self.preal[:,1]

        self.pimg[:,0] = self.PI2*(self.cf+self.f_shift)
        self.pimg[:,1] = -self.PI2*(self.cf+self.f_shift)
        self.pimg[:,2] = self.pimg[:,0]
        self.pimg[:,3] = self.pimg[:,1]
        self.pimg[:,4] = self.pimg[:,0]
        self.pimg[:,5] = self.pimg[:,1]
        
        return self.preal,self.pimg
    
#### controlers #####         
#definition of the class updater for the signal path bandpass filter
class Filter_Update: 
    def __init__(self, target,c1_coefficients,samplerate,cf,bmTaumax,bmTaumin,cohc,TauWBMax,TauWBMin):
        self.bmTaumax = bmTaumax
        self.bmTaumin = bmTaumin
        self.target=target
        self.samplerate=samplerate
        self.cf=atleast_1d(cf)
        self.cohc = cohc
        self.TauWBMax = TauWBMax
        self.TauWBMin = TauWBMin
        bmplace = 11.9 * log10(0.80 + cf / 456.0)
        self.centerfreq = 456.0*(pow(10,(bmplace+1.2)/11.9)-0.80)
        self.c1_coefficients=c1_coefficients
        self.rsigma=[]
    def __call__(self,input):  
      

        tmptauc1 = input[-1,:]
        tauc1    = self.cohc*(tmptauc1-self.bmTaumin)+self.bmTaumin
        #signal path update
        self.c1_coefficients.rsigma   = 1/tauc1-1/self.bmTaumax
#        self.target[0].filt_b,self.target[0].filt_a = self.c1_coefficients.return_coefficients() 
#        #control path update
#        tauwb = self.TauWBMax+(tauc1-self.bmTaumax)*(self.TauWBMax-self.TauWBMin)/(self.bmTaumax-self.bmTaumin)
#        [wb_gain,self.grdelay] = gain_groupdelay(1./self.samplerate,self.centerfreq,self.cf,tauwb);
#        
#        grd[n] = grdelay 
#        if ((grd[n]+n)<totalstim)
#             tmpgain[grd[n]+n] = wb_gain
#        if (tmpgain[n] == 0)
#            tmpgain[n] = lasttmpgain
#        wbgain      = tmpgain[n]
#        lasttmpgain = wbgain
        #[self.target[1].filt_b,self.target[1].filt_a] = ...
        
def IHC_transduction(x,slope,asym,sign): 
    corner    = 80
    strength  = 20.0e6/10**(corner/20)     
    xx = sign*log(1.0+strength*abs(x))*slope
    ind = x<0
    splx   = 20*log10(-x[ind]/20e-6);
    asym_t = asym-(asym-1)/(1+exp(splx/5.0));
    xx[ind] = -1/asym_t*xx[ind]
    return xx


def boltzman(x,asym,s0,s1,x1):
    shift = 1.0/(1.0+asym) # asym is the ratio of positive Max to negative Max*/
    x0    = s0*log((1.0/shift-1)/(1+exp(x1/s1)))
    out1 = 1.0/(1.0+exp(-(x-x0)/s0)*(1.0+exp(-(x-x1)/s1)))-shift
    return out1/(1-shift)


def OHC_transduction(x,taumin,taumax,asym):
    minR = 0.05*ones(len(taumin))
    R  = taumin/taumax
    ind = R<minR
    minR[ind] =  0.5*R[ind]  
    dc = (asym-1)/(asym+1.0)/2.0-minR
    R1 = R-minR
    s0 = -dc/log(R1/(1-minR))
    #This is for new nonlinearity
    x1  = abs(x)
    out = zeros_like(x)
    for ix in xrange(len(x)):
        out[ix,:] = taumax*(minR+(1.0-minR)*exp(-x1[ix,:]/s0))
        ind = out[ix,:]<taumin
        out[ix,ind] = taumin[ind]
        ind = out[ix,:]>taumax
        out[ix,ind] = taumax[ind]
    return out 


class LowPass_filter(LinearFilterbank):
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

class ZILANY(CombinedFilterbank):
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
        file="/home/bertrand/Data/MatlabProg/brian_hears/ZilanyCarney-JASAcode-2009/wbout.mat"
        X=loadmat(file,struct_as_record=False)
        wbout = Sound(X['wbout'].flatten())
        wbout.samplerate = 100*kHz

        CombinedFilterbank.__init__(self, source)
#        source = self.get_modified_source()
        
        cf = atleast_1d(cf)
        nbr_cf=len(cf)
        samplerate=source.samplerate

        parameters=set_parameters(cf,param)
#        if int(source.samplerate)!=50000:
#            warnings.warn('To use the PMFR cochlear model the sample rate should be 50kHz')
#            if not have_scikits_samplerate:
#                raise ImportError('To use the PMFR cochlear model the sample rate should be 50kHz and scikits.samplerate package is needed for resampling')               
#            #source=source.resample(50*kHz)
#            warnings.warn('The input to the PMFR cochlear model has been resampled to 50kHz')

        cohc =1 # ohc scaling factor: 1 is normal OHC function; 0 is complete OHC dysfunction
        cihc = 1 # i ihc scaling factor: 1 is normal IHC function; 0 is complete IHC dysfunction
        bmplace = 11.9 * log10(0.80 + cf / 456.0)
        centerfreq = 456.0*(pow(10,(bmplace+1.2)/11.9)-0.80)
        CAgain = minimum(60,maximum(15*ones(len(cf)),52/2*(tanh(2.2*log10(cf/600)+0.15)+1)))
        
        
        # Parameters for the control-path wideband filter =======*/
        bmorder = 3;
        Taumax,Taumin = get_tauwb(cf,CAgain,bmorder)
        taubm   = cohc*(Taumax-Taumin)+Taumin;
        ratiowb = Taumin/Taumax;
        
        #====== Parameters for the signal-path C1 filter ======*/
        bmTaumax,bmTaumin,ratiobm = get_taubm(cf,CAgain,Taumax)    
        bmTaubm  = cohc*(bmTaumax-bmTaumin)+bmTaumin;
        fcohc    = bmTaumax/bmTaubm;
        
        # Parameters for the control-path wideband filter =======*/
        wborder = 3
        TauWBMax = Taumin+0.2*(Taumax-Taumin);
        TauWBMin = TauWBMax/Taumax*Taumin;
        tauwb    = TauWBMax+(bmTaubm-bmTaumax)*(TauWBMax-TauWBMin)/(bmTaumax-bmTaumin);
        [wbgain,grdelay] = gain_groupdelay(1./self.samplerate,centerfreq,cf,tauwb)
        
        # Nonlinear asymmetry of OHC function and IHC C1 transduction function*/
        ohcasym  = 7.0    
        ihcasym  = 3.0
        
        
#        ##### Control Path ####
        control_coef = Control_Coefficients(cf, samplerate)
        [filt_b,filt_a] = control_coef.return_coefficients()
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
        LP_control=Butterworth(NL2_control, nbr_cf, 3, parameters['fc_LP_control'], btype='low')       
        
        #low pass filter for feedback to control band pass (its output will be used to control the control path)
        LP_feed_back= Butterworth(LP_control, nbr_cf, 3,parameters['fc_LP_fb'], btype='low')
#           
#        #### C1  ####
        rsigma = zeros(len(cf))
        c1_coefficients = Chirp_Coefficients(cf, bmTaumax, samplerate, rsigma,1*ones(len(cf)))
        [filt_b,filt_a] = c1_coefficients.return_coefficients()
        C1_filter = LinearFilterbank(source,filt_b,filt_a)
        C1_IHC = FunctionFilterbank(C1_filter,IHC_transduction,slope = 0.1,asym = ihcasym,sign=1) 

 
        #controlers definition
        updater=Filter_Update([C1_filter],c1_coefficients,samplerate,cf,bmTaumax,bmTaumin,cohc,TauWBMax,TauWBMin) #instantiation of the updater for the control path
        output = ControlFilterbank(C1_filter, NL2_control, [C1_filter],updater, update_interval)  #controler for the band pass filter of the control path
        self.set_output(output)

        
        #line 354