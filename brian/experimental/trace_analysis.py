'''
Analysis of voltage traces (current-clamp experiments or HH models).
'''
from brian import *
from numpy import *
#from brian.stdunits import mV
from brian.tools.io import *
from time import time
from scipy import optimize
from scipy import stats
from scipy.signal import lfilter

__all__ = ['fit_EIF', 'IV_curve', 'threshold_model', 'estimate_capacitance']

"""
TODO:
* A better fit_EIF function (conjugate gradient? Check also Badel et al., 2008. Or maybe just calculate gradient)
* Fit subthreshold kernel (least squares, see also Jolivet)
* Standard threshold methods
"""

"""
The good strategy:
* optimise for every tau for best threshold prediction
* find the tau that minimises misses
"""
def threshold_model(v, onsets=None, dt=1., tau=None):
    '''
    Fits adaptive threshold model.
    
    Model:
    tau*dvt/dt=vt0+a*[v-vi]^+
    
    tau can be fixed or found by optimization.
    
    Returns vt0, vi, a[, tau] (tau in timesteps by default)
    '''
    if onsets is None: onsets = spike_onsets(v)
    if tau is None:
        spikes = spike_mask(v, onsets)
        def f(tau):
            vt0, vi, a = threshold_model(v, onsets=onsets, dt=dt, tau=tau)
            theta = vt0 + a * lowpass(clip(v - vi, 0, inf), tau, dt=0.05)
            return mean(theta[-spikes] < v[-spikes]) # false alarm rate
        tau = float(optimize.fmin(lambda x:f(*x), 10., disp=0))
        return list(threshold_model(v, onsets=onsets, dt=dt, tau=tau)) + [tau]
    else:
        threshold = v[onsets]
        def f(vt0, vi, a, tau):
            return vt0 + a * lowpass(clip(v - vi, 0, inf), tau, dt=dt)
        return optimize.leastsq(lambda x:f(x[0], x[1], x[2], tau)[onsets] - threshold, [-55., -65., 1.])[0]

def estimate_capacitance(i, v, dt=1, guess=100.):
    '''
    Estimates capacitance from current-clamp recording
    with white noise (see Badel et al., 2008).
    
    Hint for units: if v and dt are in ms, then the capacitance has
    the same relationship to F than the current to A (pA -> pF).
    '''
    dv = diff(v) / dt
    i, v = i[:-1], v[:-1]
    mask = -spike_mask(v) # subthreshold trace
    dv, v, i = dv[mask], v[mask], i[mask]
    return optimize.fmin(lambda C:var(dv - i / C), guess, disp=0)[0]

def IV_curve(i, v, dt=1, C=None, bins=None, T=0):
    '''
    Dynamic I-V curve (see Badel et al., 2008)
    E[C*dV/dt-I]=f(V)
    T: time after spike onset to include in estimation.
    bins: bins for v (default: 20 bins between min and max).
    '''
    C = C or estimate_capacitance(i, v, dt)
    if bins is None: bins = linspace(min(v), max(v), 20)
    dv = diff(v) / dt
    v, i = v[:-1], i[:-1]
    mask = -spike_mask(v, spike_onsets(v) + T) # subthreshold trace
    dv, v, i = dv[mask], v[mask], i[mask]
    fv = i - C * dv # intrinsic current
    return array([mean(fv[(v >= vmin) & (v < vmax)]) for vmin, vmax in zip(bins[:-1], bins[1:])])

def fit_EIF(i, v, dt=1, C=None, T=0):
    '''
    Fits the exponential model of spike initiation in the
    phase plane (v,dv).
    T: time after spike onset to include in estimation.
    
    Returns gl, El, deltat, vt
    (leak conductance, leak reversal potential, slope factor, threshold)
    The result does not seem very reliable (deltat depends critically on T).
    
    Hint for units: if v is in mV, dt in ms, i in pA, then gl is in nS
    '''
    C = C or estimate_capacitance(i, v, dt)
    dv = diff(v) / dt
    v, i = v[:-1], i[:-1]
    mask = -spike_mask(v, spike_onsets(v) + T) # subthreshold trace
    dv, v, i = dv[mask], v[mask], i[mask]
    f = lambda gl, El, deltat, vt:C * dv - i - (gl * (El - v) + gl * deltat * exp((v - vt) / deltat))
    x, _ = optimize.leastsq(lambda x:f(*x), [50., -60., 3., -55.])
    return x

if __name__ == '__main__':
    filename = r"D:\My Dropbox\Neuron\Hu\recordings_Ifluct\I0_1_std_1_tau_10_sampling_20\vs.dat"
    #filename=r'D:\Anna\2010_03_0020_random_noise_200pA.atf'
    #filename2=r'D:\Anna\input_file.atf' # in pF
    from pylab import *
    #M=loadtxt(filename)
    #t,vs=M[:,0],M[:,1]
    t, vs = read_neuron_dat(filename)
    #t,vs=read_atf(filename)
    #print array(spike_duration(vs,full=True))*.05
    #vt0,vi,a,tau=threshold_model(vs,dt=0.05)
    #print vt0,vi,a,tau
    #theta=vt0+a*lowpass(clip(vs-vi,0,inf),tau,dt=0.05)
    spikes = spike_onsets(vs)
    print std(vs[spikes])
    #spikes2=spike_onsets_dv2(vs)
    #spikes3=spike_onsets_dv3(vs)
    plot(t, vs, 'k')
    #plot(t,theta,'b')
    plot(t[spikes], vs[spikes], '.r')
    #plot(t[spikes2],vs[spikes2],'.g')
    #plot(t[spikes3],vs[spikes3],'.k')
    show()
