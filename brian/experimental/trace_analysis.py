'''
Analysis of voltage traces (current-clamp experiments or HH models).
'''
from numpy import *
#from brian.stdunits import mV
from brian.utils.io import *
from time import time
from scipy import optimize
from scipy import stats

__all__=['find_spike_criterion','spike_peaks','spike_onsets','find_onset_criterion',
         'slope_threshold','vm_threshold','spike_shape']

# TODO: I-V curve, slopefactor, subthreshold kernel, time constant, threshold optimisation, remove spikes

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
    vc=vc or find_spike_criterion(v)
    criterion=criterion or find_onset_criterion(v,vc=vc)
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
    vc=vc or find_spike_criterion(v)
    return float(optimize.fmin(lambda x:std(v[spike_onsets(v,x,vc)]),guess,disp=0))

def spike_shape(v,onsets=None,before=100,after=100):
    '''
    Spike shape (before peaks). Aligned on spike onset by default
    (to align on peaks, just pass onsets=peaks).
    
    onsets: spike onset times
    before: number of timesteps before onset
    after: number of timesteps after onset
    '''
    if onsets is None: onsets=spike_onsets(v)
    shape=zeros(after+before)
    for i in onsets:
        v0=v[max(0,i-before):i+after]
        shape[len(shape)-len(v0):]+=v0
    return shape/len(onsets)

def vm_threshold(v,onsets=None,T=None):
    '''
    Average membrane potential before spike threshold (T steps).
    '''
    if onsets is None: onsets=spike_onsets(v)
    l=[]
    for i in onsets:
        l.append(mean(v[max(0,i-T):i]))
    return array(l)

def slope_threshold(v,onsets=None,T=None):
    '''
    Slope of membrane potential before spike threshold (T steps).
    '''
    if onsets is None: onsets=spike_onsets(v)
    l=[]
    for i in onsets:
        v0=v[max(0,i-T):i]
        M=len(v0)
        x=arange(M)-M+1
        slope=sum((v0-v[i])*x)/sum(x**2)
        l.append(slope)
    return array(l)

def estimate_capacitance(i,v,dt=1,guess=1.):
    '''
    Estimates capacitance from current-clamp recording
    with white noise (see Badel et al., 2008).
    '''
    dv=diff(v)/dt
    i=i[:-1]
    return optimize.fmin(lambda C:var(dv-i/C),guess,disp=0)[0]

"""
def estimate_area(i,v,dt=1,guess=1.):
    return estimate_capacitance(i,v,dt,guess)/(0.9*uF/cm**2)
"""

def fit_EIF(i,v,dt=1,C=None):
    '''
    Fits the exponential model of spike initiation in the
    phase plane (v,dv).
    '''
    C=C or estimate_capacitance(i,v,dt)
    # Remove spikes?
    dv=diff(v)/dt
    v=v[:-1]
    i=i[:-1]
    f=lambda gl,El,deltat,vt:C*dv-i-(gl*(El-v)+gl*deltat*exp((v-vt)/deltat))
    #df=lambda gl,El,deltat,vt:gl*(v-gl*exp((v-vt)/deltat))
    x,_=optimize.leastsq(lambda x:f(*x),[1./80.,-60.,3.,-55.])
    #gl,El,deltat,vt=x
    return x

def remove_spikes(v,spikes=None,T=100):
    if spikes is None:
        spikes=spike_peaks(v)
    ind=(v!=1e9)
    for i in spikes:
        ind[i:i+100]=False
    return ind

"""
def IV_curve(i,v,dt=1):
    '''
    Dynamic I-V curve (see Badel et al., 2008)
    E[C*dV/dt-I]=f(V)
    NB: maybe be better to have direct fit to exponential?
    '''
    C=estimate_capacitance(i,v,dt)
"""

if __name__=='__main__':
    #path=r'D:\My Dropbox\Neuron\Hu\recordings_Ifluct\I0_07_std_02_tau_10_sampling_20\\'
    filename=r'D:\Anna\2010_03_0020_random_noise_200pA.atf'
    filename2=r'D:\Anna\input_file.atf' # in pF
    from pylab import *
    M=loadtxt(filename)
    t,vs=M[:,0],M[:,1]
    M2=loadtxt(filename2)
    _,i=M2[:,0],M2[:,1]
    t=t[:len(i)]
    vs=vs[:len(i)]
    print "Data loaded"
    ind=remove_spikes(vs)
    vs=vs[ind]
    i=i[ind]
    C=estimate_capacitance(i,vs,0.05*1e-3)# in nF
    print C
    dv=diff(vs)/(0.05*1e-3)
    figure()
    plot(vs[:-1],C*dv-i[:-1])
    figure()
    plot(vs[:-1],C*dv)
    print fit_EIF(i,vs,0.05*1e-3)
    #exit()
    #shape=spike_shape(vs,before=100,after=100)
    #spikes=spike_onsets(vs)
    #plot(vs[:-1],diff(vs))
    #vm=vm_threshold(vs,spikes,T=200)
    #ISI=diff(spikes)
    #subplot(211)
    #plot(vm,vs[spikes],'.')
    #subplot(212)
    #plot(ISI,vs[spikes[1:]],'.')
    #slope=slope_threshold(vs,spikes,T=200)
    #plot(slope,vs[spikes],'.')
    #plot(shape)
    #hist(vs[spikes])
    #plot(t,vs)
    #plot(t[spikes],vs[spikes],'.r')
    show()
