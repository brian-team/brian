'''
Analysis of voltage traces (current-clamp experiments or HH models).
'''
from numpy import *
from brian.stdunits import mV

def spike_times(v):
    '''
    Returns the indexes of spike times.
    '''
    return ((v[1:]>0) & (v[:-1]<0)).nonzero()[0]

def find_spike_criterion(v):
    '''
    This is a rather complex method to determine above which voltage vs
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

def find_spike_criterion2(v):
    '''
    This is a rather complex method to determine above which voltage vs
    one should consider that a spike is produced.
    It must satisfy:
      if v(t)>vs and v(t-dt)<vs, then v must increase and the next peak
      must be at least 20*mV
      (and possibly: go back below vs within a sufficiently short time)
    Choose the lowest possible vs.
    '''
    pass

if __name__=='__main__':
    path=r'D:\My Dropbox\Neuron\Hu\recordings_Ifluct\I0_07_std_02_tau_10_sampling_20\\'
    t,vs=read_neuron_dat(path+'vs.dat')
    print find_spike_criterion(vs)
