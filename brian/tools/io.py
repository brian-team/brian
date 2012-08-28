'''
Input/output utility functions
'''
import numpy as np
import os
from brian.stdunits import *
from brian.units import *

__all__ = ['read_neuron_dat', 'read_atf', 'load_aer']

# General readers

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

# Spikes saving (AER readers)

def load_multiple_aer(filename, check_sorted = False, relative_time = False, directory = '.'):
    '''
    Loads multiple AER files listed in the file.
    '''
    f=open(filename,'rb')
    line = f.readline()
    res = []
    line = line.strip('\n')
    while not line == '':
        res.append(load_aer(os.path.join(directory, line), check_sorted = check_sorted, reinit_time = relative_time))
        line = f.readline()
    f.close()
    return res


def load_aer(filename, 
             check_sorted = False, 
             reinit_time = False):
    '''
    Loads Address Event Representation (AER) data files for use in
    Brian. Files contain spikes as a binary representation of an
    ``address`` (i.e. neuron identifier) and a timestamp.

    This function returns two arrays, an array of addresses (neuron indices) and an array of spike times (in second).

    Note: For index files (that point to multiple .(ae)dat files, typically aeidx files) it
    will return a list containing tuples (addr, time) as for single files.
    
    Usage:
    
    ids, times = load_aer('/path/to/file.aedat')
    
    Keyword Arguments:

    ``reinit_time`` If True, sets the first spike time to zero and all others relative to that one.
    
    ``check_sorted`` If True, checks if timestamps are sorted, and sorts them if necessary.
    
    Example use:
    
    To use the spikes recorded in the AER file ``filename`` in a Brian ``NeuronGroup``, one should do:

    addr, timestamp =  load_AER(filename, reinit_time = True)
    G = AERSpikeGeneratorGroup((addr, timestamps))

    An example script can be found in examples/misc/spikes_io.py

    
    '''
    # This loading fun is inspired by the following Matlab script:
    # http://jaer.svn.sourceforge.net/viewvc/jaer/trunk/host/matlab/loadaerdat.m?revision=2001&content-type=text%2Fplain
    
    
    # Figure out extension, check filename, ...
    l = filename.split('.')
    ext = l[-1].strip('\n')
    filename = filename.strip('\n')
    directory = os.path.dirname(filename)
    
    if ext == 'aeidx':
        #AER data points to different AER files
        return load_multiple_AER(filename, 
                                 check_sorted = check_sorted, 
                                 relative_time = reinit_time, 
                                 directory = directory)
    elif not (ext == 'dat' or ext == 'aedat'):
        raise ValueError('Wrong extension for AER data, should be dat, or aedat, it was '+ext)
    
    f=open(filename,'rb')

    # Load the encoding parameters
    # version of AER
    version=1 # default (if not found in the file)
    # value of dt
    dt = 1e-6 # default (if not found in the file)
    
    # skip header and look overrident values for dt/version
    line = f.readline()
    while len(line) == 0 or line[0] == '#':
        if line[:9] == "#!AER-DAT":
            version = int(float(line[9:-1]))
        if line[:21] == '# Timestamps tick is ':
            dt = eval(line[21:])
#            print 'recognized dt = %.4f second' % dt
        line = f.readline()
    line += f.read()
    f.close()

    # Load the files
    if version==1:
        '''
        Format is: sequence of (addr = 2 bytes,timestamp = 4 bytes)
        Number format is big endian ('>')
        '''
        ## This commented paragraph is the non-vectorized version
        #nevents=len(line)/6
        #for n in range(nevents):
        #    events.append(unpack('>HI',line[n*6:(n+1)*6])) # address,timestamp
        x=np.fromstring(line, dtype=np.int16) # or uint16?
        x=x.reshape((len(x)/3,3))
        addr=x[:,0].newbyteorder('>')
        timestamp=x[:,1:].copy()
        timestamp.dtype=int32
        timestamp=timestamp.newbyteorder('>').flatten()
    else: # i.e. version==2
        '''
        Format is: sequence of (addr = 4 bytes,timestamp = 4 bytes)
        Number format is big endian ('>')
        '''
        ## This commented paragraph is the non-vectorized version
        #nevents=len(line)/8
        #for n in range(nevents):
        #    events.append(unpack('>II',line[n*8:(n+1)*8])) # address,timestamp
        x = np.fromstring(line, dtype=np.int32).newbyteorder('>')
        addr = x[::2]
        if len(addr) == len(x[1::2]):
            timestamp = x[1::2]
        else:
            # alternative fallback:
            #timestamp = x[1::2]
            #print len(x)
            raise IOError("""Corrupted AER file, timestamps and addresses don't have the same lengths.""")
    
    # Sorts the events if necessary
    if check_sorted: 
        if any(np.diff(timestamp)<0): # not sorted
            ind = np.argsort(timestamp)
            addr,timestamp = addr[ind],timestamp[ind]

    # Set first spike time to 0 if required
    if reinit_time:
        t0 = min(timestamp)
        timestamp -= t0

    # Check for negative timestamps (i dont remember why?)
    if (timestamp<0).all():
        raise ValueError("""AER file contains (only!) negative timestamps, \
consider using reinit_time = True""")
    
    return addr, timestamp * dt

