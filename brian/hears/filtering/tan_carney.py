import numpy as np
from numpy import pi
import scipy.signal as signal
import warnings

from brian.clock import Clock
from brian.stdunits import Hz, ms
from brian.threshold import PoissonThreshold
from brian.reset import CustomRefractoriness
from brian.neurongroup import NeuronGroup
from brian.network import network_operation

from brian.hears.filtering.filterbank import (FunctionFilterbank,
                                              ControlFilterbank,
                                              CombinedFilterbank,
                                              RestructureFilterbank)
from brian.hears.filtering.linearfilterbank import LinearFilterbank
from brian.hears.filtering.filterbankgroup import FilterbankGroup


__all__=['TanCarney', 'MiddleEar', 'ZhangSynapse']

class MiddleEar(LinearFilterbank):
    '''
    Implements the middle ear model from Tan & Carney (2003) (linear filter
    with two pole pairs and one double zero). The gain is normalized for the
    response of the analog filter at 1000Hz as in the model of Tan & Carney
    (their actual C code does however result in a slightly different
    normalization, the difference in overall level is about 0.33dB (to get
    exactly the same output as in their model, set the `gain` parameter to
    0.962512703689).

    Tan, Q., and L. H. Carney.
    "A Phenomenological Model for the Responses of Auditory-nerve Fibers.
    II. Nonlinear Tuning with a Frequency Glide".
    The Journal of the Acoustical Society of America 114 (2003): 2007.
    '''
    def __init__(self, source, gain=1, **kwds):
        # Automatically duplicate mono input to fit the desired output shape
        gain = np.atleast_1d(gain)
        if len(gain) != source.nchannels and len(gain) != 1:
            if source.nchannels != 1:
                raise ValueError('Can only automatically duplicate source '
                                 'channels for mono sources, use '
                                 'RestructureFilterbank.')
            source = RestructureFilterbank(source, len(gain))
        samplerate = source.samplerate
        zeros = np.array([-200, -200])
        poles = np.array([-250 + 400j, -250 - 400j,
                          -2000 + 6000j, -2000 - 6000j])
        # use an arbitrary gain here, will be normalized afterwards
        b, a = signal.zpk2tf(zeros, poles * 2 * np.pi, 1.5e9)
        # normalize the response at 1000Hz (of the analog filter)
        resp = np.abs(signal.freqs(b, a, [1000*2*np.pi])[1])  # response magnitude
        b /= resp
        bd, ad = signal.bilinear(b, a, samplerate)
        bd = (np.tile(bd, (source.nchannels, 1)).T * gain).T
        ad = np.tile(ad, (source.nchannels, 1))
        LinearFilterbank.__init__(self, source, bd, ad, **kwds)


class ZhangSynapseSpikes(NeuronGroup):
    '''
    The spike-generating Poisson process (with absolute and relative
    refractoriness) of an IHC-AN synapse according to the Zhang et al. (2001)
    model. The `source` has to have a state variable `s`, representing the
    firing rate (e.g. the class `ZhangSynapseRate`).
    
    The `n_per_channel` argument can be used to generate multiple spike trains
    for every channel of the source group.
    '''
    def __init__(self, source, n_per_channel=1, params=None):
        params = ZhangSynapse._get_parameters(params)
        c_0, c_1 = params['c_0'], params['c_1']
        s_0, s_1 = params['s_0'], params['s_1']
        R_A = params['R_A']
        eqs =  '''
        # time-varying discharge rate, input into this model
        s : Hz
        
        # discharge-history effect (Equation 20 in differential equation form)        
        H = c_0*e_0 + c_1*e_1 : 1
        de_0/dt = -e_0/s_0    : 1
        de_1/dt = -e_1/s_1    : 1

        # final time-varying discharge rate for the Poisson process, equation 19
        R = s * (1 - H) : Hz
        '''
        
        def reset_func(P, spikes):
            P.e_0[spikes] = 1.0
            P.e_1[spikes] = 1.0

        # make sure that the s value is first updated in
        # ZhangSynapseRate, then this NeuronGroup is
        # updated
        clock=Clock(dt=source.clock.dt, t=source.clock.t,
                    order=source.clock.order + 1)
        
        @network_operation(clock=clock, when='start')
        def distribute_input():
            self.s[:] = source.s.repeat(n_per_channel)
        
        NeuronGroup.__init__(self, len(source) * n_per_channel,
                             model=eqs,
                             threshold=PoissonThreshold('R'),
                             reset=CustomRefractoriness(resetfun=reset_func,
                                                        period=R_A),
                             clock=clock
                             )
        
        self.contained_objects += [distribute_input]


class ZhangSynapse(ZhangSynapseSpikes):
    '''
    A `FilterbankGroup` that represents an IHC-AN synapse according to the
    Zhang et al. (2001) model. The `source` should be a filterbank, producing
    V_ihc (e.g. `TanCarney`). `CF` specifies the characteristic frequencies of
    the AN fibers. `params` overwrites any parameters values given in the
    publication.

    The group emits spikes according to a time-varying Poisson process with
    absolute and relative refractoriness (probability of spiking is given by
    state variable ``R``). The continuous probability of spiking without
    refractoriness is available in the state variable ``s``. 

    The `n_per_channel` argument can be used to generate multiple spike trains
    for every channel.

    If all you need is the state variable ``s``, you can use the class
    `ZhangSynapseRate` instead which does not simulate the spike-generating
    Poisson process.

    For details see:
    Zhang, X., M. G. Heinz, I. C. Bruce, and L. H. Carney.
    "A Phenomenological Model for the Responses of Auditory-nerve Fibers:
    I. Nonlinear Tuning with Compression and Suppression".
    The Journal of the Acoustical Society of America 109 (2001): 648.
    '''
    
    @staticmethod
    def _get_parameters(params=None):
        # Default values for parameters from table 1, Zhang et al. 2001
        default_params = {'spont': 50*Hz,
                          # In the example C code, this is used (with comment: "read Frank's cmpa.c")
                          'A_SS': 130*Hz, 
                          'tau_ST': 60*ms,
                          'tau_R': 2*ms,
                          'A_RST': 6,
                          'PTS': 8.627,
                          'P_Imax': 0.6,
                          'c_0': 0.5,
                          'c_1': 0.5,
                          'R_A': 0.75*ms,
                          's_0': 1*ms,
                          's_1': 12.5*ms}
        if params is None:
            return default_params
    
        for param, value in params.iteritems():
            if not param in default_params:
                raise KeyError(('"%s" is not a valid parameter, '
                                'has to be one of: %s') % (param,
                                                           str(default_params.keys())))
            default_params[param] = value
        
        return default_params     
    
    def __init__(self, source, CF, n_per_channel=1, params=None):
        params = ZhangSynapse._get_parameters(params)
        
        rate_model = ZhangSynapseRate(source, CF, params)        
        ZhangSynapseSpikes.__init__(self, rate_model, n_per_channel, params)
        
        self.contained_objects += [rate_model]


class ZhangSynapseRate(FilterbankGroup):
    '''
    A `FilterbankGroup` that represents an IHC-AN synapse according to the
    Zhang et al. (2001) model, see `ZhangSynapse` for details. This class does
    not actually generate any spikes, it only simulates the time-varying
    firing rate (not taking refractory effects into account) `s`.
    '''

    def __init__(self, source, CF, params=None):
        
        params = ZhangSynapse._get_parameters(params)
        
        spont = params['spont']
        A_SS = params['A_SS']
        tau_ST = params['tau_ST']
        tau_R = params['tau_R']
        A_RST = params['A_RST']
        PTS = params['PTS']
        P_Imax = params['P_Imax']
        c_0 = params['c_0']
        c_1 = params['c_1']
        R_A = params['R_A']
        s_0 = params['s_0']
        s_1 = params['s_1']                  
        
        # Equations A1-A5 of Zhang et al. 2001
        A_ON  = PTS * A_SS  # onset rate
        A_R = (A_ON - A_SS) * A_RST / (1 + A_RST)  # rapid response amplitude
        A_ST = A_ON - A_SS - A_R  # short-term response amplitude
        P_rest = P_Imax * spont / A_ON  # resting permeability
        C_G = spont * (A_ON - spont) / (A_ON*P_rest*(1 - spont/A_SS))  # global concentration
    
        # Equations A6 (intermediate parameters for store volume computation)
        gamma_1 = C_G / spont
        gamma_2 = C_G / A_SS
        kappa_1 = -1 / tau_R
        kappa_2 = -1 / tau_ST
    
        # Equations A7-A9 (immediate volume)
        V_I0 = (1 - P_Imax/P_rest)/(gamma_1*((A_R*(kappa_1-kappa_2)/(C_G*P_Imax)) +
                                             kappa_2/(P_rest*gamma_1) - kappa_2/(P_Imax*gamma_2)))
        V_I1 = (1 - P_Imax/P_rest)/(gamma_1*((A_ST*(kappa_2-kappa_1)/(C_G*P_Imax)) +
                                             kappa_1/(P_rest*gamma_1) - kappa_1/(P_Imax*gamma_2)))
        V_I = 0.5 * (V_I0 + V_I1)
    
        # Equations A10 (other intermediate parameters)
        alpha = gamma_2 / (kappa_1*kappa_2)
        beta = -(kappa_1 + kappa_2) * alpha
        theta_1 = alpha * P_Imax/V_I
        theta_2 = V_I/P_Imax
        theta_3 = gamma_2 - 1/P_Imax
    
        # Equations A11-A12 (local and global permeabilities)
        P_L = ((beta - theta_2*theta_3)/theta_1 - 1) * P_Imax
        P_G = 1 / (theta_3 - 1/P_L)
    
        # Equations A13-A15
        V_L = theta_1*P_L*P_G  # local volume
        C_Irest = spont/P_rest  # resting value of immediate concentration
        C_Lrest = C_Irest*(P_rest + P_L)/P_L  # local concentration
    
        # Equation 18 with A16 and A17
        p_1 = P_rest / np.log(2)
    
        eqs = '''
        # input into the Synapse
        V_ihc : 1

        # CF in Hz
        CF_param : 1

        # Equation A17 (using an expression based on the spontaneous rate instead of 18.54, based on the C code)
        V_sat = 20.0*(spont + 1*Hz)/(spont + 5*Hz)*P_Imax*((V_sat2 > 1.5)*(V_sat2 - 1.5) + 1.5) : 1
        V_sat2 = 2 + 3*np.log10(CF_param / 1000.0) : 1

        # Equation 17
        P_I_exponent = p_2 * V_ihc : 1
        # avoid infinity values
        P_I = p_1 * np.clip(np.log(1 + np.exp(P_I_exponent)), -np.inf, np.abs(P_I_exponent) + 1): 1
    
        # Following Equation A16 (p_2 is the same as P_ST)
        p_2_exponent = np.log(2) * V_sat / P_rest : 1
        p_2 = np.clip(np.log(np.exp(p_2_exponent) - 1), -np.inf, np.abs(p_2_exponent)) : 1

        # Equation A18-A19
        # Concentration in the stores (as differential instead of difference equation)
        dC_I/dt = (-P_I*C_I + P_L*(C_L - C_I))/V_I : Hz
        dC_L/dt = (-P_L*(C_L - C_I) + P_G*(C_G - C_L))/V_I : Hz

        # time-varying discharge rate (ignoring refractory effects), equation A20
        s = C_I * P_I : Hz
        '''
    
        FilterbankGroup.__init__(self, source, 'V_ihc', model=eqs)
        self.CF_param = CF
        self.C_I = C_Irest
        self.C_L = C_Lrest
        

def set_parameters(cf,param):
    
    parameters=dict()
    parameters['fc_LP_control']=800 #Hz
    parameters['fc_LP_fb']=500 #Hz
    parameters['fp1']=1.0854*cf-106.0034
    parameters['ta']=10**(np.log10(cf)*1.0230 + 0.1607)
    parameters['tb']=10**(np.log10(cf)*1.4292 - 1.1550) - 1000
    parameters['gain80']=10**(np.log10(cf)*0.5732 + 1.5220)
    parameters['rgain']=10**( np.log10(cf)*0.4 + 1.9)
    parameters['average_control']=0.3357
    parameters['zero_r']= np.array(-10**( np.log10(cf)*1.5-0.9 ))   
        
    if param: 
        if not isinstance(param, dict): 
            raise TypeError('given parameters must be a dict')
        for key in param.keys():
            if key != 'nlgain' and not parameters.has_key(key):
                raise KeyError(key + ' is invalid key entry for given parameters')
            parameters[key] = param[key]

    parameters['nlgain']= (parameters['gain80'] - parameters['rgain'])/parameters['average_control']
    return parameters


class Control_Coefficients:
    
    def __init__(self,cf,samplerate):
        self.cf = cf
        self.PI2 = 2.*3.14159265358979
        self.nch=len(cf)
        self.fs_bilinear = 2.0*samplerate#*ones(self.nch)
#        self.fs_bilinear =tile(self.fs_bilinear.reshape(self.nch,-1),3)
        self.x_cf=11.9*np.log10(0.8+cf/456);
        self.f_shift=(pow(10,((self.x_cf+1.2)/11.9))-0.8)*456-cf
        self.wbw=cf/4.0
        self.filt_a = np.zeros((len(cf),3,5), order='F') #8 5
#        self.filt_a[:,0,:] = 1
        self.filt_b = np.zeros((len(cf),3,5), order='F')
        self.control_signal = 0
        self.preal = np.zeros((self.nch,6))
        self.pimg = np.zeros((self.nch,6))
        self.preal,self.pimg = self.analog_poles()
        
    def return_coefficients(self,control_signal):
        self.wbw=-(self.preal[:,0] - control_signal)/self.PI2
        
        self.gain_norm_bp=((self.PI2**2
                            * np.sqrt(self.wbw**2 + self.f_shift**2)
                            * np.sqrt((2*self.cf+self.f_shift)**2 + self.wbw**2)
                            )**3)/np.sqrt(self.PI2**2*self.cf**2)#       
        iord = [1,3,5]  
        
        preal = self.preal[:,iord]-control_signal.T  #actually control_signal is the same for the three channels
        
        temp=(self.fs_bilinear-(preal))**2 + self.pimg[:,iord]**2    
        
        self.filt_a[:,0,0:3] = 1.
        self.filt_a[:,1,0:3]= -2*(self.fs_bilinear**2-(preal)**2-self.pimg[:,iord]**2)/temp            
        self.filt_a[:,2,0:3] = ((self.fs_bilinear+(preal))**2+self.pimg[:,iord]**2)/temp
        self.filt_b[:,0,0:3] = 1./temp
        self.filt_b[:,1,0:3] = 2./temp  
        self.filt_b[:,2,0:3] = 1./temp
        
        self.filt_a[:,0,3] = 1.
        self.filt_a[:,1,3]= 1.      ## changed  from 1 to 0
        self.filt_a[:,2,3] = 0.
        self.filt_b[:,0,3] = self.fs_bilinear
        self.filt_b[:,1,3] = -self.fs_bilinear
        self.filt_b[:,2,3] = 0
        
#        self.filt_b[:,:,3] = self.gain_norm_bp*self.filt_b[:,:,3]  
        
        self.filt_a[:,0,4] = 1.
        self.filt_b[:,0,4] = self.gain_norm_bp
  
        
        return self.filt_b,self.filt_a

    def analog_poles(self):
        self.preal[:,0] = -self.PI2*self.wbw  #that should be -, actually there are never used
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
        self.t = 0
        self.cf = cf
        self.PI2 = 2*3.14159265358979
        self.nch=len(cf)
        self.fs_bilinear = 2.0*samplerate#*ones(self.nch)
        
        self.order_of_pole = 20             
        self.half_order_pole = self.order_of_pole/2
        self.order_of_zero  = self.half_order_pole

        self.filt_a = np.zeros((len(cf),3,11), order='F')
        self.filt_b = np.zeros((len(cf),3,11), order='F')
        
        self.preal = np.zeros((self.nch,20))
        self.pimg = np.zeros((self.nch,20))
        
        self.control_signal = 0
        
        self.rgain =parameters['rgain']# 10**(log10(cf)*0.4 + 1.9)
        self.fp1=parameters['fp1']#1.0854*cf-106.0034
        self.ta= parameters['ta']#    10**(log10(cf)*1.0230 + 0.1607)
        self.tb= parameters['tb']#    10**(log10(cf)*1.4292 - 1.1550) - 1000
        self.zeroa = parameters['zero_r']# -10**(log10(cf)*1.5-0.9 )  
        
        self.zeroamat = np.tile(self.zeroa.reshape(self.nch,-1),10)
        self.preal,self.pimg = self.analog_poles(0)

        
        self.cfmat = np.tile(self.cf.reshape(self.nch,-1),20)  
        self.gain_norm = np.sqrt(np.prod((2*pi*self.cfmat-self.pimg[:,0:20])**2+self.preal[:,0:20]**2,axis=1))

        self.gain_norm = self.gain_norm /(np.sqrt((2*pi*self.cf)**2+self.zeroa**2))**self.order_of_zero
        
    def return_coefficients(self,control_signal):
        self.preal,self.pimg = self.analog_poles(control_signal)
        
        iord = np.arange(2,22,2)-1
        temp=(self.fs_bilinear-self.preal[:,iord])**2 + self.pimg[:,iord]**2
        self.filt_a[:,0,:10] = 1
        self.filt_a[:,1,:10] = -2*(self.fs_bilinear**2-self.preal[:,iord]**2-self.pimg[:,iord]**2)/temp            
        self.filt_a[:,2,:10] = ((self.fs_bilinear+self.preal[:,iord])**2+self.pimg[:,iord]**2)/temp
        
        self.filt_b[:,0,:10] = (-self.zeroamat+self.fs_bilinear)/temp
        self.filt_b[:,1,:10] = (-2*self.zeroamat)/temp
        self.filt_b[:,2,:10] = (-self.zeroamat-self.fs_bilinear)/temp 
        
        self.filt_a[:,0,10] = 1.
        self.filt_b[:,0,10] = self.gain_norm/3.
        
        return self.filt_b,self.filt_a

    def analog_poles(self,control_signal):
        aa = -self.rgain-control_signal
        aa[aa>=0] = 100
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
    def __call__(self, input):
        reshaped_input = input[-1,:].reshape(1,-1)
        self.target.filt_b,self.target.filt_a = self.coef.return_coefficients(reshaped_input) 
        self.param.append(self.coef.control_signal)


class LowPass_IHC(LinearFilterbank):
    def __init__(self,source,cf,fc,gain,order): 
        nch = len(cf)
        TWOPI = 2*pi
        self.samplerate =  source.samplerate
        c = 2.0 * self.samplerate
        c1LP = ( c/Hz - TWOPI*fc ) / ( c/Hz + TWOPI*fc )
        c2LP = TWOPI*fc/Hz / (TWOPI*fc + c/Hz)
        
        b_temp = np.array([c2LP,c2LP])
        a_temp = np.array([1,-c1LP])
        
        filt_b = np.tile(b_temp.reshape([2,1]),[nch,1,order])               
        filt_a = np.tile(a_temp.reshape([2,1]),[nch,1,order]) 
        filt_b[:,:,0] = filt_b[:,:,0]*gain

        LinearFilterbank.__init__(self, source, filt_b, filt_a)


class LowPass_filter(LinearFilterbank):
    def __init__(self,source,cf,fc,gain,order):
        nch = len(cf)
        TWOPI = 2*pi
        self.samplerate =  source.samplerate
        c = 2.0 * self.samplerate
        c1LP = ( c/Hz - TWOPI*fc ) 
        
        b_temp = np.array([1,1])/ ( c/Hz + TWOPI*fc )
        a_temp = np.array([1,-c1LP/ ( c/Hz + TWOPI*fc )])
        
        filt_b = np.tile(b_temp.reshape([2,1]),[nch,1,order])               
        filt_a = np.tile(a_temp.reshape([2,1]),[nch,1,order]) 
        filt_b[:,:,order-1] = filt_b[:,:,order-1]*gain

        LinearFilterbank.__init__(self, source, filt_b, filt_a)


def saturation_fc(x,A0=1,B=1,C=1,D=1):
    ind = x>=0
    x[ind]=A0*np.log(x[ind]*B+1.0)   
    ind = x<0
    dtemp = (-x[ind])**C
    tempA = -A0*(dtemp+D)/(3*dtemp+D)
    x[ind]=tempA*np.log(abs(x[ind])*B+1.0)

    return x


class TanCarneyIHC(CombinedFilterbank):
    
    def __init__(self, source, cf):
        CombinedFilterbank.__init__(self, source)
        source = self.get_modified_source()
        
        ## Saturation
        saturation = FunctionFilterbank(source, saturation_fc,
                                        A0=0.1, B=2000, C=1.74, D=6.87e-9)
        ## low pass IHC
        ihc = LowPass_IHC(saturation, cf, 3800, 1, 7)
        self.set_output(ihc)


class TanCarneyControl(CombinedFilterbank):
    def __init__(self, source, cf, update_interval, param=None):
        CombinedFilterbank.__init__(self, source)
        source = self.get_modified_source()       
        cf = np.atleast_1d(cf)
        samplerate=source.samplerate
        parameters = set_parameters(cf, param)
        ##### Control Path ####
        # band pass filter
        control_coef = Control_Coefficients(cf, samplerate)
        [filt_b,filt_a] = control_coef.return_coefficients(np.zeros((1,len(cf))))        
        BP_control = LinearFilterbank(source,filt_b,filt_a)
                
        # first non linearity of control path
        Acp,Bcp,Ccp=100.,2.5,0.60 
        func_NL1_control=lambda x:np.sign(x)*Bcp*np.log(1.+Acp*abs(x)**Ccp)
        NL1_control=FunctionFilterbank(BP_control,func_NL1_control)
                
        # second non linearity of control path
        asym,s0,x1,s1=7.,8.,5.,3. 
        shift = 1./(1.+asym)
        x0 = s0*np.log((1.0/shift-1)/(1+np.exp(x1/s1)))
        func_NL2_control=lambda x:(1.0/(1.0+np.exp(-(x-x0)/s0)*(1.0+np.exp(-(x-x1)/s1)))-shift)*parameters['nlgain']
        NL2_control=FunctionFilterbank(NL1_control,func_NL2_control)

        #control low pass filter (its output will be used to control the signal path)
        gain_lp_con = (2*pi*parameters['fc_LP_control'])**3*1.5
        LP_control = LowPass_filter(NL2_control,cf,parameters['fc_LP_control'],gain_lp_con,3)   
        #low pass filter for feedback to control band pass (its output will be used to control the control path)
        gain_lp_fb = parameters['fc_LP_fb']*2*pi*10
        LP_feed_back = LowPass_filter(LP_control,cf,parameters['fc_LP_fb'],gain_lp_fb,1)
        
        updater = Filter_Update(BP_control, control_coef) #instantiation of the updater for the control path
        output = ControlFilterbank(LP_control, LP_feed_back, BP_control,
                                   updater, update_interval)  #controler for the band pass filter of the control path
                 
        self.set_output(output)


class TanCarneySignal(CombinedFilterbank):    
    def __init__(self, source, cf, update_interval, param=None):

        CombinedFilterbank.__init__(self, source)
        source = self.get_modified_source()       
        cf = np.atleast_1d(cf)
        parameters = set_parameters(cf, param)
        samplerate=source.samplerate

        if int(source.samplerate)!=50000:
            warnings.warn('To use the TanCarney cochlear model the sample rate should be 50kHz')
        
        # band pass filter
        signal_coef = Signal_Coefficients(cf, samplerate,parameters)
        [filt_b,filt_a] = signal_coef.return_coefficients(np.zeros((1,len(cf))))
        BP_signal = LinearFilterbank(source,filt_b,filt_a)
        
        control_output = TanCarneyControl(source, cf, update_interval, parameters)
        
        updater = Filter_Update(BP_signal, signal_coef) #instantiation of the updater for the signal path
        output = ControlFilterbank(BP_signal, control_output, BP_signal,
                                   updater, update_interval)  #controler for the band pass filter of the signal path

        self.set_output(output)


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
        
    def __init__(self, source, cf, update_interval=1, param=None):
        CombinedFilterbank.__init__(self, source)
        source = self.get_modified_source()       
        cf = np.atleast_1d(cf)

        parameters=set_parameters(cf,param)        
        
        signal = TanCarneySignal(source, cf, update_interval, parameters)
        ihc = TanCarneyIHC(signal, cf)
        
        self.set_output(ihc)
