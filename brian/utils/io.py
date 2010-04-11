'''
Input/output utility functions
'''
import numpy as np

def read_neuron_dat(name):
    '''
    Reads a Neuron vector file (.dat).
    
    Returns vector of times, vector of values
    '''
    f=open(name)
    f.readline(),f.readline() # skip first two lines
    M=np.loadtxt(f)
    f.close()
    return M[:,0],M[:,1]

if __name__=='__main__':
    from pylab import *
    t,v=read_neuron_dat(r"D:\My Dropbox\Neuron\Hu\recordings_Ifluct\I0_05_std_1_tau_10_sampling_100\vs.dat")
    plot(t[:50000],v[:50000])
    show()
