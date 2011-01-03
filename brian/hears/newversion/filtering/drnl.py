from brian import *
set_global_preferences(usenewbrianhears=True,
                       useweave=False)
from brian.hears import *
from filterbank import Filterbank,FunctionFilterbank
from filterbanklibrary import *

def set_parameters(cf,type,given_param):
    
    parameters=dict()
    parameters['stape_scale']=0.00014 
    parameters['order_linear']=3
    parameters['order_nonlinear']=3

    if type=='ginnea_pig':
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
        
    if given_param: 
        if not isinstance(given_param, dict): 
            raise Error('given parameters must be a dict')
        for key in given_param.keys():
            if not parameters.has_key(key):
                raise Exception(key + ' is invalid key entry for given parameters')
            parameters[key] = given_param[key]
    return parameters
    
class DRNL(Filterbank):
    '''
    
    Implementation of the dual resonance nonlinear (DRNL) filter.

    
    The entire pathway consists of the sum of a linear and a nonlinear pathway
    
    The linear path consists of a bandpass function (second order gammatone), a low pass function,
    and a gain/attenuation factor, g, in a cascade
    
    The nonlinear path is  a cascade consisting of a bandpass function, a
    compression function, a second bandpass function, and a low
    pass function, in that order.
    
    The parameters are those fitted for human
    from Lopez-Paveda, E. and Meddis, R.., A human nonlinear cochlear filterbank, JASA 2001
    
    '''
    
    def __new__(cls, source,cf,type='human',given_param={}):
        
        cf = atleast_1d(cf)
        nbr_cf=len(cf)
        parameters=set_parameters(cf,type,given_param)
        
        #conversion to stape velocity (which are the units needed for the further centres)
        source=source*parameters['stape_scale'] 
        
        #### Linear Pathway ####
        #bandpass filter (second order  gammatone filter)
        cf_linear=10**(parameters['cf_lin_p0']+parameters['cf_lin_m']*log10(cf))
        bandwidth_linear=10**(parameters['bw_lin_p0']+parameters['bw_lin_m']*log10(cf))
        gammatone=ApproximateGammatoneFilterbank(source, cf_linear, bandwidth_linear, order=parameters['order_linear'])
        #linear gain
        g=10**(parameters['g_p0']+parameters['g_m']*log10(cf))
        func_gain=lambda x:g*x
        gain= FunctionFilterbank(gammatone,func_gain)
        #low pass filter(cascade of 4 second order lowpass butterworth filters)
        cutoff_frequencies_linear=10**(parameters['lp_lin_cutoff_p0']+parameters['lp_lin_cutoff_m']*log10(cf))
        order_lowpass_linear=2
        lp_l=LowPassFilterbank(gain,cutoff_frequencies_linear)
        lowpass_linear=CascadeFilterbank(gain,lp_l,4)
        
        #### Nonlinear Pathway ####
        #bandpass filter (third order gammatone filters)
        cf_nonlinear=10**(parameters['cf_nl_p0']+parameters['cf_nl_m']*log10(cf))
        bandwidth_nonlinear=10**(parameters['bw_nl_p0']+parameters['bw_nl_m']*log10(cf))
        bandpass_nonlinear1=ApproximateGammatoneFilterbank(source, cf_nonlinear, bandwidth_nonlinear, order=parameters['order_nonlinear'])
        #compression (linear at low level, compress at high level)
        a=10**(parameters['a_p0']+parameters['a_m']*log10(cf))  #linear gain
        b=10**(parameters['b_p0']+parameters['b_m']*log10(cf))  
        v=10**(parameters['c_p0']+parameters['c_m']*log10(cf))#compression exponent
        func_compression=lambda x:sign(x)*minimum(a*abs(x),b*abs(x)**v)
        compression=FunctionFilterbank(bandpass_nonlinear1,  func_compression)
        #bandpass filter (third order gammatone filters)
        bandpass_nonlinear2=ApproximateGammatoneFilterbank(compression, cf_nonlinear, bandwidth_nonlinear, order=parameters['order_nonlinear'])
        #low pass filter
        cutoff_frequencies_nonlinear=10**(parameters['lp_nl_cutoff_p0']+parameters['lp_nl_cutoff_m']*log10(cf))
        order_lowpass_nonlinear=2
        lp_nl=LowPassFilterbank(bandpass_nonlinear2,cutoff_frequencies_nonlinear)
        lowpass_nonlinear=CascadeFilterbank(bandpass_nonlinear2,lp_nl,3)
        #adding the two pathways
        dnrl_filter=lowpass_linear+lowpass_nonlinear
        return dnrl_filter

    
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