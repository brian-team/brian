'''
An implementation of the IHC-AN synapse from

Zhang, X., M. G. Heinz, I. C. Bruce, and L. H. Carney.
"A Phenomenological Model for the Responses of Auditory-nerve Fibers:
I. Nonlinear Tuning with Compression and Suppression".
The Journal of the Acoustical Society of America 109 (2001): 648.

'''

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from brian.units import second, check_units
from brian.stdunits import kHz, Hz, ms
from brian.network import Network
from brian.monitor import StateMonitor, SpikeMonitor
from brian.globalprefs import set_global_preferences
from brian.threshold import PoissonThreshold
from brian.reset import CustomRefractoriness

#set_global_preferences(useweave=True)
from brian.hears import (Sound, LinearFilterbank, FilterbankGroup, get_samplerate,
                         set_default_samplerate, tone, click, silence, dB, TanCarney)

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


class ZhangSynapse(FilterbankGroup):

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
    
        for param, value in params.iter_values():
            if not param in default_params:
                raise KeyError(('"%s" is not a valid parameter, '
                                'has to be one of: %s') % (param,
                                                           str(default_params.keys())))
            default_params[param] = value
        
        return default_params 

    def __init__(self, source, CF, params=None):
        '''
        Create a `FilterbankGroup` that represents a synapse according to the
        Zhang et al. (2001) model. The `source` should be a filterbank, producing
        V_ihc (e.g. `TanCarney`). `CF` specifies the characteristic frequencies of
        the AN fibers. `params` overwrites any parameters values given in the
        publication.
    
        The group emits spikes according to a time-varying Poisson process with
        absolute and relative refractoriness (probability of spiking is given by
        state variable ``R``). The continuous probability of spiking without
        refractoriness is available in the state variable ``s``.
    
        For details see Zhang et al. (2001).
        '''
        
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
    
        FilterbankGroup.__init__(self, source, 'V_ihc',
                                 model=eqs,
                                 threshold=PoissonThreshold('R'),
                                 reset=CustomRefractoriness(resetfun=reset_func,
                                                            period=R_A)
                                 )
        self.CF_param = CF
        self.C_I = C_Irest
        self.C_L = C_Lrest


if __name__ == '__main__':
    '''
    Fig. 1 and 3 (spking output without spiking/refractory period) should
    reproduce the output of the AN3_test_tone.m and AN3_test_click.m
    scripts, available in the code accompanying the paper Tan & Carney (2003).
    This matlab code is available from
    http://www.urmc.rochester.edu/labs/Carney-Lab/publications/auditory-models.cfm

    Tan, Q., and L. H. Carney.
    "A Phenomenological Model for the Responses of Auditory-nerve Fibers.
    II. Nonlinear Tuning with a Frequency Glide".
    The Journal of the Acoustical Society of America 114 (2003): 2007.
    '''
    set_default_samplerate(50*kHz)
    sample_length = 1 / get_samplerate(None)
    cf = 1000 * Hz
    
    print 'Testing click response'
    duration = 25*ms    
    levels = [40, 60, 80, 100, 120]
    # a click of two samples
    tones = Sound([Sound.sequence([click(sample_length*2, peak=level*dB),
                                   silence(duration=duration - sample_length)])
               for level in levels])
    ihc = TanCarney(MiddleEar(tones), [cf] * len(levels), update_interval=1)
    syn = ZhangSynapse(ihc, cf)
    s_mon = StateMonitor(syn, 's', record=True, clock=syn.clock)
    R_mon = StateMonitor(syn, 'R', record=True, clock=syn.clock)
    spike_mon = SpikeMonitor(syn)
    net = Network(syn, s_mon, R_mon, spike_mon)
    net.run(duration * 1.5)
    for idx, level in enumerate(levels):
        plt.figure(1)
        plt.subplot(len(levels), 1, idx + 1)
        plt.plot(s_mon.times / ms, s_mon[idx])
        plt.xlim(0, 25)
        plt.xlabel('Time (msec)')
        plt.ylabel('Sp/sec')
        plt.text(15, np.nanmax(s_mon[idx])/2., 'Peak SPL=%s SPL' % str(level*dB));
        if idx == 0:
            plt.title('Click responses')

        plt.figure(2)
        plt.subplot(len(levels), 1, idx + 1)
        plt.plot(R_mon.times / ms, R_mon[idx])
        plt.xlabel('Time (msec)')
        plt.xlabel('Time (msec)')
        plt.text(15, np.nanmax(s_mon[idx])/2., 'Peak SPL=%s SPL' % str(level*dB));
        if idx == 0:
            plt.title('Click responses (with spikes and refractoriness)')
        plt.plot(spike_mon.spiketimes[idx] / ms,
             np.ones(len(spike_mon.spiketimes[idx])) * np.nanmax(R_mon[idx]), 'rx')

    print 'Testing tone response'
    duration = 60*ms    
    levels = [0, 20, 40, 60, 80]
    tones = Sound([Sound.sequence([tone(cf, duration).atlevel(level*dB).ramp(when='both',
                                                                             duration=10*ms,
                                                                             inplace=False),
                                   silence(duration=duration/2)])
                   for level in levels])
    ihc = TanCarney(MiddleEar(tones), [cf] * len(levels), update_interval=1)
    syn = ZhangSynapse(ihc, cf)
    s_mon = StateMonitor(syn, 's', record=True, clock=syn.clock)
    R_mon = StateMonitor(syn, 'R', record=True, clock=syn.clock)
    spike_mon = SpikeMonitor(syn)
    net = Network(syn, s_mon, R_mon, spike_mon)
    net.run(duration * 1.5)
    for idx, level in enumerate(levels):
        plt.figure(3)
        plt.subplot(len(levels), 1, idx + 1)
        plt.plot(s_mon.times / ms, s_mon[idx])
        plt.xlim(0, 120)
        plt.xlabel('Time (msec)')
        plt.ylabel('Sp/sec')
        plt.text(1.25 * duration/ms, np.nanmax(s_mon[idx])/2., '%s SPL' % str(level*dB));
        if idx == 0:
            plt.title('CF=%.0f Hz - Response to Tone at CF' % cf)

        plt.figure(4)
        plt.subplot(len(levels), 1, idx + 1)
        plt.plot(R_mon.times / ms, R_mon[idx])
        plt.xlabel('Time (msec)')
        plt.xlabel('Time (msec)')
        plt.text(1.25 * duration/ms, np.nanmax(R_mon[idx])/2., '%s SPL' % str(level*dB));
        if idx == 0:
            plt.title('CF=%.0f Hz - Response to Tone at CF (with spikes and refractoriness)' % cf)
        plt.plot(spike_mon.spiketimes[idx] / ms,
             np.ones(len(spike_mon.spiketimes[idx])) * np.nanmax(R_mon[idx]), 'rx')

    plt.show()
