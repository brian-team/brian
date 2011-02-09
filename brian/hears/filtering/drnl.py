from brian import *
from filterbank import Filterbank, FunctionFilterbank, CombinedFilterbank
from filterbanklibrary import *

__all__ = ['DRNL']

def set_parameters(cf,type,param):
    
    parameters=dict()
    parameters['stape_scale']=0.00014 
    parameters['order_linear']=3
    parameters['order_nonlinear']=3

    if type=='guinea pig':
        parameters['cf_lin_p0']=0.339
        parameters['cf_lin_m']=0.339
        parameters['bw_lin_p0']=1.3
        parameters['bw_lin_m']=0.53
        parameters['cf_nl_p0']=0
        parameters['cf_nl_m']=1.
        parameters['bw_nl_p0']=0.8
        parameters['bw_nl_m']=0.58
        parameters['a_p0']=1.87
        parameters['a_m']=0.45
        parameters['b_p0']=-5.65
        parameters['b_m']=0.875
        parameters['c_p0']=-1.
        parameters['c_m']=0
        parameters['g_p0']=5.68
        parameters['g_m']=-0.97
        parameters['lp_lin_cutoff_p0']=0.339
        parameters['lp_lin_cutoff_m']=0.339
        parameters['lp_nl_cutoff_p0']=0
        parameters['lp_nl_cutoff_m']=1.
        
    elif type=='human':
        parameters['cf_lin_p0']=-0.067
        parameters['cf_lin_m']=1.016
        parameters['bw_lin_p0']=0.037
        parameters['bw_lin_m']=0.785
        parameters['cf_nl_p0']=-0.052
        parameters['cf_nl_m']=1.016
        parameters['bw_nl_p0']=-0.031
        parameters['bw_nl_m']=0.774
        parameters['a_p0']=1.402
        parameters['a_m']=0.819
        parameters['b_p0']=1.619
        parameters['b_m']=-0.818  
        parameters['c_p0']=-0.602
        parameters['c_m']=0
        parameters['g_p0']=4.2
        parameters['g_m']=0.48
        parameters['lp_lin_cutoff_p0']=-0.067
        parameters['lp_lin_cutoff_m']=1.016
        parameters['lp_nl_cutoff_p0']=-0.052
        parameters['lp_nl_cutoff_m']=1.016
        
    if param: 
        if not isinstance(param, dict): 
            raise Error('given parameters must be a dict')
        for key in param.keys():
            if not parameters.has_key(key):
                raise Exception(key + ' is invalid key entry for given parameters')
            parameters[key] = param[key]
    return parameters
    
class DRNL(CombinedFilterbank):
    r'''
    Implementation of the dual resonance nonlinear (DRNL) filter
    as described in Lopez-Paveda, E. and Meddis, R.,
    "A human nonlinear cochlear filterbank", JASA 2001.
    
    The entire pathway consists of the sum of a linear and a nonlinear pathway.
    
    The linear path consists of a bank of bandpass filters (second order
    gammatone), a low pass function, and a gain/attenuation factor, g, in a
    cascade.
    
    The nonlinear path is a cascade consisting of a bank of gammatone filters, a
    compression function, a second bank of gammatone filters, and a low
    pass function, in that order.

    Initialised with arguments:
    
    ``source``
        Source of the cochlear model.
        
    ``cf``
        List or array of center frequencies.
        
    ``type``
        defines the parameters set corresponding to a certain fit. It can be
        either:
        
        ``type='human'`` 
            The parameters come from Lopez-Paveda, E. and Meddis, R.., "A human
            nonlinear cochlear filterbank", JASA 2001.
        
        ``type ='guinea pig'``
            The parameters come from Summer et al., "A nonlinear filter-bank
            model of the guinea-pig cochlear nerve: Rate responses", JASA 2003.
        
    ``param``
        Dictionary used to overwrite the default parameters given in the
        original papers. 
    
    The possible parameters to change and their default values for humans (see 
    Lopez-Paveda, E. and Meddis, R.,"A human nonlinear cochlear filterbank",
    JASA 2001. for notation) are::
      
      param['stape_scale']=0.00014 
      param['order_linear']=3 
      param['order_nonlinear']=3 
    
    from there on the parameters are given in the form
    :math:`x=10^{\mathrm{p0}+m\log_{10}(\mathrm{cf})}` where
    ``cf`` is the center frequency::
    
      param['cf_lin_p0']=-0.067
      param['cf_lin_m']=1.016
      param['bw_lin_p0']=0.037
      param['bw_lin_m']=0.785
      param['cf_nl_p0']=-0.052
      param['cf_nl_m']=1.016
      param['bw_nl_p0']=-0.031
      param['bw_nl_m']=0.774
      param['a_p0']=1.402
      param['a_m']=0.819
      param['b_p0']=1.619
      param['b_m']=-0.818  
      param['c_p0']=-0.602
      param['c_m']=0
      param['g_p0']=4.2
      param['g_m']=0.48
      param['lp_lin_cutoff_p0']=-0.067
      param['lp_lin_cutoff_m']=1.016
      param['lp_nl_cutoff_p0']=-0.052
      param['lp_nl_cutoff_m']=1.016   
    '''
    
    def __init__(self, source, cf, type='human', param={}):

        CombinedFilterbank.__init__(self, source)
        source = self.get_modified_source()
        
        cf = atleast_1d(cf)
        nbr_cf=len(cf)
        parameters=set_parameters(cf,type,param)
        
        #conversion to stape velocity (which are the units needed for the further centres)
        source=source*parameters['stape_scale'] 
        
        #### Linear Pathway ####
        #bandpass filter (second order  gammatone filter)
        cf_linear=10**(parameters['cf_lin_p0']+parameters['cf_lin_m']*log10(cf))
        bandwidth_linear=10**(parameters['bw_lin_p0']+parameters['bw_lin_m']*log10(cf))
        gammatone=ApproximateGammatone(source, cf_linear, bandwidth_linear, order=parameters['order_linear'])
        #linear gain
        g=10**(parameters['g_p0']+parameters['g_m']*log10(cf))
        func_gain=lambda x:g*x
        gain= FunctionFilterbank(gammatone,func_gain)
        #low pass filter(cascade of 4 second order lowpass butterworth filters)
        cutoff_frequencies_linear=10**(parameters['lp_lin_cutoff_p0']+parameters['lp_lin_cutoff_m']*log10(cf))
        order_lowpass_linear=2
        lp_l=LowPass(gain,cutoff_frequencies_linear)
        lowpass_linear=Cascade(gain,lp_l,4)
        
        #### Nonlinear Pathway ####
        #bandpass filter (third order gammatone filters)
        cf_nonlinear=10**(parameters['cf_nl_p0']+parameters['cf_nl_m']*log10(cf))
        bandwidth_nonlinear=10**(parameters['bw_nl_p0']+parameters['bw_nl_m']*log10(cf))
        bandpass_nonlinear1=ApproximateGammatone(source, cf_nonlinear, bandwidth_nonlinear, order=parameters['order_nonlinear'])
        #compression (linear at low level, compress at high level)
        a=10**(parameters['a_p0']+parameters['a_m']*log10(cf))  #linear gain
        b=10**(parameters['b_p0']+parameters['b_m']*log10(cf))  
        v=10**(parameters['c_p0']+parameters['c_m']*log10(cf))#compression exponent
        func_compression=lambda x:sign(x)*minimum(a*abs(x),b*abs(x)**v)
        compression=FunctionFilterbank(bandpass_nonlinear1,  func_compression)
        #bandpass filter (third order gammatone filters)
        bandpass_nonlinear2=ApproximateGammatone(compression, cf_nonlinear, bandwidth_nonlinear, order=parameters['order_nonlinear'])
        #low pass filter
        cutoff_frequencies_nonlinear=10**(parameters['lp_nl_cutoff_p0']+parameters['lp_nl_cutoff_m']*log10(cf))
        order_lowpass_nonlinear=2
        lp_nl=LowPass(bandpass_nonlinear2,cutoff_frequencies_nonlinear)
        lowpass_nonlinear=Cascade(bandpass_nonlinear2,lp_nl,3)
        #adding the two pathways
        drnl_filter=lowpass_linear+lowpass_nonlinear

        self.set_output(drnl_filter)

    
if __name__ == '__main__':        
    
    from brian import *
    set_global_preferences(usenewbrianhears=True,
                           useweave=False)
    from brian.hears import *

    dBlevel=60*dB  # dB level in rms dB SPL
    sound=Sound.load('/home/bertrand/Data/Toolboxes/AIM2006-1.40/Sounds/aimmat.wav')
    samplerate=sound.samplerate
    sound=sound.atlevel(dBlevel)
    
    
    simulation_duration=len(sound)/samplerate
    
    nbr_center_frequencies=50
    center_frequencies=log_space(100*Hz, 1000*Hz, nbr_center_frequencies)
    dnrl_filter=DRNL(sound,center_frequencies)
    dnrl_filter.buffer_init()
    dnrl=dnrl_filter.buffer_fetch(0, len(sound))