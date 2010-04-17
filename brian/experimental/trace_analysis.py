'''
Analysis of voltage traces (current-clamp experiments or HH models).
'''
from numpy import *
#from brian.stdunits import mV
from brian.utils.io import *
from time import time
from scipy.optimize import fmin

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
    dv=diff(v)
    sign_changes=((dv[1:]*dv[:-1])<=0).nonzero()[0]
    vc=v[sign_changes+1]
    i=argmax(diff(vc))
    return .5*(vc[i]+vc[i+1])

def spike_peaks(v,vc=None):
    '''
    Returns the indexes of spike peaks.
    vc is the spike criterion (voltage above which we consider we have a spike)
    '''
    # Possibly: add refractory criterion
    vc=vc or find_spike_criterion(v)
    dv=diff(v)
    spikes=((v[1:]>vc) & (v[:-1]<vc)).nonzero()[0]
    peaks=[]
    for i in range(len(spikes)-1):
        peaks.append(spikes[i]+(dv[spikes[i]:spikes[i+1]]<=0).nonzero()[0][0])
    peaks.append(spikes[-1]+(dv[spikes[-1]:]<=0).nonzero()[0][0])
    return array(peaks)

def spike_onsets(v,criterion=None,vc=None):
    '''
    Returns the indexes of spike onsets.
    vc is the spike criterion (voltage above which we consider we have a spike).
    First derivative criterion (dv>criterion).
    '''
    peaks=spike_peaks(v,vc)
    dv=diff(v)
    previous_i=0
    j=0
    l=[]
    for i in peaks:
        # Find peak of derivative (alternatively: last sign change of d2v, i.e. last local peak)
        inflexion=previous_i+argmax(dv[previous_i:i])
        j+=max((dv[j:inflexion]<criterion).nonzero()[0])+1
        l.append(j)
        previous_i=i
    return array(l)

def find_onset_criterion(v,guess=0.1,vc=None):
    '''
    Finds the best criterion on dv/dt to determine spike onsets,
    based on minimum threshold variability.
    '''
    return float(fmin(lambda x:std(v[spike_onsets(v,x,vc)]),guess,disp=0))

def slope_threshold(v):
    # 1. find onsets
    # 2. linear regression on slopes
    # 3. optimization to find best correlated time for slopes
    pass

if __name__=='__main__':
    path=r'D:\My Dropbox\Neuron\Hu\recordings_Ifluct\I0_07_std_02_tau_10_sampling_20\\'
    t,vs=read_neuron_dat(path+'vs.dat')
    print find_spike_criterion(vs)
    sp=spike_peaks(vs)
    c=find_onset_criterion(vs)
    spikes=spike_onsets(vs,c)
    print c/.02,std(vs[spikes])
    from pylab import *
    hist(vs[spikes])
    show()
    #plot(t,vs)
    #plot(t[spikes],vs[spikes],'.r')
    #show()
