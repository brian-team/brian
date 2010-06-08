'''
Input/output utility functions
'''
import numpy as np

__all__ = ['read_neuron_dat', 'read_atf']

def read_neuron_dat(name):
    '''
    Reads a Neuron vector file (.dat).
    
    Returns vector of times, vector of values
    '''
    f = open(name)
    f.readline(), f.readline() # skip first two lines
    M = np.loadtxt(f)
    f.close()
    return M[:, 0], M[:, 1]

def read_atf(name):
    '''
    Reads an Axon ATF file (.atf).
    
    Returns vector of times, vector of values
    '''
    f = open(name)
    f.readline()
    n = int(f.readline().split()[0]) # skip first two lines
    for _ in range(n + 1):
        f.readline()
    M = np.loadtxt(f)
    f.close()
    return M[:, 0], M[:, 1]

if __name__ == '__main__':
    from pylab import *
    t, v = read_neuron_dat(r"D:\My Dropbox\Neuron\Hu\recordings_Ifluct\I0_05_std_1_tau_10_sampling_100\vs.dat")
    #t,v=read_atf(r"/home/bertrand/Data/Measurements/Anna/input_file.atf")
    plot(t[:50000], v[:50000])
    show()
