"""
Analysis of voltage traces.

Mainly about analysis of spike shapes.
"""
from numpy import *
from scipy import optimize
from scipy.signal import lfilter

__all__ = ['find_spike_criterion', 'spike_peaks', 'spike_onsets', 'find_onset_criterion',
         'slope_threshold', 'vm_threshold', 'spike_shape', 'spike_duration', 'reset_potential',
         'spike_mask', 'lowpass', 'spike_onsets_dv2', 'spike_onsets_dv3']

def lowpass(x, tau, dt=1.):
    """
    Low-pass filters x(t) with time constant tau.
    """
    a = exp(-dt / tau)
    return lfilter([1. - a], [1., -a], x)

def spike_duration(v, onsets=None, full=False):
    '''
    Average spike duration.
    Default: time from onset to next minimum.
    If full is True:
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
    i = argmax(diff(vc)) # I think there is a mistake, I should sort vc first
    return .5 * (vc[i] + vc[i + 1])

def spike_peaks(v, vc=None):
    '''
    Returns the indexes of spike peaks.
    vc is the spike criterion (voltage above which we consider we have a spike)
    '''
    # Possibly: add refractory criterion
    if vc is None: vc = find_spike_criterion(v)
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
    if vc is None: vc = find_spike_criterion(v)
    if criterion is None: criterion = find_onset_criterion(v, vc=vc)
    peaks = spike_peaks(v, vc)
    dv = diff(v)
    d2v = diff(dv)
    previous_i = 0
    j = 0
    l = []

    for i in peaks:
        # Find last peak of derivative (commented: point where derivative is largest)
        # inflexion = previous_i + argmax(dv[previous_i:i])
        inflexion=where(d2v[previous_i:i-2]*d2v[previous_i+1:i-1]<0)[0][-1]+2+previous_i
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
    if vc is None: vc = find_spike_criterion(v)
    peaks = spike_peaks(v, vc)
    d2v = diff(diff(v))
    d3v = diff(d2v) # I'm guessing you have to shift v by 1/2 per differentiation
    j = 0
    l = []
    previous_i=0
    for i in peaks:
        # Find peak of derivative
        inflexion=where(d2v[previous_i:i-1]*d2v[previous_i+1:i]<0)[0][-1]+2+previous_i
        j += max(((d3v[j:inflexion - 1] > 0) & (d3v[j + 1:inflexion] < 0)).nonzero()[0]) # +2?
        l.append(j)
        previous_i=i
    return array(l)

def spike_onsets_dv3(v, vc=None):
    '''
    Returns the indexes of spike onsets.
    vc is the spike criterion (voltage above which we consider we have a spike).
    Maximum of 3rd derivative.
    DOESN'T SEEM GOOD
    '''
    if vc is None: vc = find_spike_criterion(v)
    peaks = spike_peaks(v, vc)
    dv4 = diff(diff(diff(diff(v))))
    j = 0
    l = []
    for i in peaks:
        # Find peak of derivative (alternatively: last sign change of d2v, i.e. last local peak)
        j += max(((dv4[j:i - 1] > 0) & (dv4[j + 1:i] < 0)).nonzero()[0]) + 3
        l.append(j)
    return array(l)

def find_onset_criterion(v, guess=0.0001, vc=None):
    '''
    Finds the best criterion on dv/dt to determine spike onsets,
    based on minimum threshold variability.
    '''
    if vc is None: vc = find_spike_criterion(v)
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
    Returns all slopes as an array.
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
