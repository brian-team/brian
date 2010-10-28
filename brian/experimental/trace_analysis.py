'''
Analysis of voltage traces (current-clamp experiments or HH models).
'''
from numpy import *
#from brian.stdunits import mV
from brian.tools.io import *
from time import time
from scipy import optimize
from scipy import stats
from scipy.signal import lfilter

__all__ = ['find_spike_criterion', 'spike_peaks', 'spike_onsets', 'find_onset_criterion',
         'slope_threshold', 'vm_threshold', 'spike_shape', 'spike_duration', 'reset_potential',
         'spike_mask', 'fit_EIF', 'IV_curve', 'threshold_model', 'lowpass', 'estimate_capacitance',
         'spike_onsets_dv2', 'spike_onsets_dv3']

"""
TODO:
* A better fit_EIF function (conjugate gradient? Check also Badel et al., 2008. Or maybe just calculate gradient)
* Fit subthreshold kernel (least squares, see also Jolivet)
* Standard threshold methods
"""

def lowpass(x, tau, dt=1.):
    """
    Low-pass filter x(t) with time constant tau.
    """
    a = exp(-dt / tau)
    return lfilter([1. - a], [1., -a], x)

def spike_duration(v, onsets=None, full=False):
    '''
    Average spike duration.
    Default: time from onset to next minimum.
    If full:
    * Time from onset to peak
    * Time from onset down to same value (spike width)
    * Total duration from onset to next minimum
    * Standard deviations for these 3 values
    '''
    if onsets is None: onsets = spike_onsets(v)
    dv = diff(v)
    total_duration = []
    time_to_peak = []
    spike_width = []
    for i, spike in enumerate(onsets):
        if i == len(onsets) - 1:
            next_spike = len(dv)
        else:
            next_spike = onsets[i + 1]
        total_duration.append(((dv[spike:next_spike - 1] <= 0) & (dv[spike + 1:next_spike] > 0)).argmax())
        time_to_peak.append((dv[spike:next_spike] <= 0).argmax())
        spike_width.append((v[spike + 1:next_spike] <= v[spike]).argmax())
    if full:
        return mean(time_to_peak), mean(spike_width), mean(total_duration), \
               std(time_to_peak), std(spike_width), std(total_duration)
    else:
        return mean(total_duration)

def reset_potential(v, peaks=None, full=False):
    '''
    Average reset potential, calculated as next minimum after spike peak.
    If full is True, also returns the standard deviation.
    '''
    if peaks is None: peaks = spike_peaks(v)
    dv = diff(v)
    reset = []
    for i, spike in enumerate(peaks):
        if i == len(peaks) - 1:
            next_spike = len(dv)
        else:
            next_spike = peaks[i + 1]
        reset.append(v[spike + ((dv[spike:next_spike - 1] <= 0) & (dv[spike + 1:next_spike] > 0)).argmax() + 1])
    if full:
        return mean(reset), std(reset)
    else:
        return mean(reset)

def find_spike_criterion(v):
    '''
    This is a rather complex method to determine above which voltage vc
    one should consider that a spike is produced.
    We look in phase space (v,dv/dt), at the horizontal axis dv/dt=0.
    We look for a voltage for which the voltage cannot be still.
    Algorithm: find the largest interval of voltages for which there is no
    sign change of dv/dt, and pick the middle.
    '''
    # Rather: find the lowest maximum?
    dv = diff(v)
    sign_changes = ((dv[1:] * dv[:-1]) <= 0).nonzero()[0]
    vc = v[sign_changes + 1]
    i = argmax(diff(vc))
    return .5 * (vc[i] + vc[i + 1])

def spike_peaks(v, vc=None):
    '''
    Returns the indexes of spike peaks.
    vc is the spike criterion (voltage above which we consider we have a spike)
    '''
    # Possibly: add refractory criterion
    vc = vc or find_spike_criterion(v)
    dv = diff(v)
    spikes = ((v[1:] > vc) & (v[:-1] < vc)).nonzero()[0]
    peaks = []
    if len(spikes) > 0:
        for i in range(len(spikes) - 1):
            peaks.append(spikes[i] + (dv[spikes[i]:spikes[i + 1]] <= 0).nonzero()[0][0])
        decreasing = (dv[spikes[-1]:] <= 0).nonzero()[0]
        if len(decreasing) > 0:
            peaks.append(spikes[-1] + decreasing[0])
        else:
            peaks.append(len(dv)) # last element (maybe should be deleted?)
    return array(peaks)

def spike_onsets(v, criterion=None, vc=None):
    '''
    Returns the indexes of spike onsets.
    vc is the spike criterion (voltage above which we consider we have a spike).
    First derivative criterion (dv>criterion).
    '''
    vc = vc or find_spike_criterion(v)
    criterion = criterion or find_onset_criterion(v, vc=vc)
    peaks = spike_peaks(v, vc)
    dv = diff(v)
    previous_i = 0
    j = 0
    l = []
    for i in peaks:
        # Find peak of derivative (alternatively: last sign change of d2v, i.e. last local peak)
        inflexion = previous_i + argmax(dv[previous_i:i])
        j += max((dv[j:inflexion] < criterion).nonzero()[0]) + 1
        l.append(j)
        previous_i = i
    return array(l)

def spike_onsets_dv2(v, vc=None):
    '''
    Returns the indexes of spike onsets.
    vc is the spike criterion (voltage above which we consider we have a spike).
    Maximum of 2nd derivative.
    DOESN'T SEEM GOOD
    '''
    vc = vc or find_spike_criterion(v)
    peaks = spike_peaks(v, vc)
    dv3 = diff(diff(diff(v))) # I'm guessing you have to shift v by 1/2 per differentiation
    j = 0
    l = []
    for i in peaks:
        # Find peak of derivative (alternatively: last sign change of d2v, i.e. last local peak)
        j += max(((dv3[j:i - 1] > 0) & (dv3[j + 1:i] < 0)).nonzero()[0]) + 2
        l.append(j)
    return array(l)

def spike_onsets_dv3(v, vc=None):
    '''
    Returns the indexes of spike onsets.
    vc is the spike criterion (voltage above which we consider we have a spike).
    Maximum of 3rd derivative.
    DOESN'T SEEM GOOD
    '''
    vc = vc or find_spike_criterion(v)
    peaks = spike_peaks(v, vc)
    dv4 = diff(diff(diff(diff(v))))
    j = 0
    l = []
    for i in peaks:
        # Find peak of derivative (alternatively: last sign change of d2v, i.e. last local peak)
        j += max(((dv4[j:i - 1] > 0) & (dv4[j + 1:i] < 0)).nonzero()[0]) + 3
        l.append(j)
    return array(l)

def find_onset_criterion(v, guess=0.1, vc=None):
    '''
    Finds the best criterion on dv/dt to determine spike onsets,
    based on minimum threshold variability.
    '''
    vc = vc or find_spike_criterion(v)
    return float(optimize.fmin(lambda x:std(v[spike_onsets(v, x, vc)]), guess, disp=0))

def spike_shape(v, onsets=None, before=100, after=100):
    '''
    Spike shape (before peaks). Aligned on spike onset by default
    (to align on peaks, just pass onsets=peaks).
    
    onsets: spike onset times
    before: number of timesteps before onset
    after: number of timesteps after onset
    '''
    if onsets is None: onsets = spike_onsets(v)
    shape = zeros(after + before)
    for i in onsets:
        v0 = v[max(0, i - before):i + after]
        shape[len(shape) - len(v0):] += v0
    return shape / len(onsets)

def vm_threshold(v, onsets=None, T=None):
    '''
    Average membrane potential before spike threshold (T steps).
    '''
    if onsets is None: onsets = spike_onsets(v)
    l = []
    for i in onsets:
        l.append(mean(v[max(0, i - T):i]))
    return array(l)

def slope_threshold(v, onsets=None, T=None):
    '''
    Slope of membrane potential before spike threshold (T steps).
    Returns all slopes as a list.
    '''
    if onsets is None: onsets = spike_onsets(v)
    l = []
    for i in onsets:
        v0 = v[max(0, i - T):i]
        M = len(v0)
        x = arange(M) - M + 1
        slope = sum((v0 - v[i]) * x) / sum(x ** 2)
        l.append(slope)
    return array(l)

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

def spike_mask(v, spikes=None, T=None):
    '''
    Returns an array of booleans which are True in spikes.
    spikes: starting points of spikes (default: onsets)
    T: duration (default: next minimum)
    
    Ex:
      v=v[spike_mask(v)] # only spikes
      v=v[-spike_mask(v)] # subthreshold trace
    '''
    if spikes is None: spikes = spike_onsets(v)
    ind = (v == 1e9)
    if T is None:
        dv = diff(v)
        for i, spike in enumerate(spikes):
            if i == len(spikes) - 1: next_spike = len(dv)
            else: next_spike = spikes[i + 1]
            T = ((dv[spike:next_spike - 1] <= 0) & (dv[spike + 1:next_spike] > 0)).argmax()
            ind[spike:spike + T + 1] = True
    else: # fixed duration
        for i in spikes:
            ind[i:i + T] = True
    return ind

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
