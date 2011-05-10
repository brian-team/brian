"""
Module to deal with the AER (Address Event Representation) format.

Current state:
* load_AER seems fine
* extract_DVS_event is probably fine too, but maybe should be in a "chip" module?
"""
#from struct import *
from numpy import *
from brian.directcontrol import SpikeGeneratorGroup
from brian.units import *
from brian.neurongroup import *
from brian.directcontrol import SpikeGeneratorThreshold
from brian.clock import guess_clock
from brian.stateupdater import *

import os

__all__=['load_AER','extract_DVS_event', 'AERSpikeGeneratorGroup', 'load_multiple_AER']

class AERSpikeGeneratorGroup(NeuronGroup):
    '''
    This class loads AER data files and puts them in a SpikeGeneratorGroup for use in Brian.
    (one can find sample data files in http://sourceforge.net/apps/trac/jaer/wiki/AER%20data)
    
    This can load any AER files that is supported by load_AER, apart from index (.aeidx) files that point to multiple data files. Check the documentation for load_AER for that.
    
    Sample usage:
    Gin = AERSpikeGeneratorGroup('/path/to/file/samplefile.dat')
    '''
    def __init__(self, data, clock = None, timeunit = 1*usecond):
        if isinstance(data, str):
            ext = data.split('.').strip('\n')
            if ext == 'aeidx':
                raise ValueError('Cannot create a single AERSpikeGeneratorGroup with aeidx files. Consider using load_AER first and manually create multiple AERSpikeGeneratorGroups.')
            else:
                data = load_AER(data, relative_time = True, check_sorted = True)
        addr, timestamps = data
        self.tmax = max(timestamps)*timeunit
        N = max(addr) + 1
        clock = guess_clock(clock)
        threshold = FastDCThreshold(addr, timestamps*timeunit, dt = clock.dt)
        NeuronGroup.__init__(self, N, model = LazyStateUpdater(), threshold = threshold, clock = clock)
    
    @property
    def maxtime(self):
        if not isinstance(self.tmax, Quantity):
            return self.tmax*second
        return self.tmax

class FastDCThreshold(SpikeGeneratorThreshold):
    '''
    Implementing dan's idea for fast Direct Control Threshold, works like a charm.
    '''
    def __init__(self, addr, timestamps, dt = None):
        self.set_offsets(addr, timestamps, dt = dt)
        
    def set_offsets(self, I, T, dt = 1000):
        # Convert times into integers
        T = array(T/dt, dtype=int)
        # Put them into order
        # We use a field array to sort first by time and then by neuron index
        spikes = zeros(len(I), dtype=[('t', int), ('i', int)])
        spikes['t'] = T
        spikes['i'] = I
        spikes.sort(order=('t', 'i'))
        T = spikes['t']
        self.I = spikes['i']
        # Now for each timestep, we find the corresponding segment of I with
        # the spike indices for that timestep.
        # The idea of offsets is that the segment offsets[t]:offsets[t+1]
        # should give the spikes with time t, i.e. T[offsets[t]:offsets[t+1]]
        # should all be equal to t, and so then later we can return
        # I[offsets[t]:offsets[t+1]] at time t. It might take a bit of thinking
        # to see why this works. Since T is sorted, and bincount[i] returns the
        # number of elements of T equal to i, then j=cumsum(bincount(T))[t]
        # gives the first index in T where T[j]=t.
        self.offsets = hstack((0, cumsum(bincount(T))))
    
    def __call__(self, P):
        t = P.clock.t
        dt = P.clock.dt
        curt = float(t)
        t = int(round(t/dt))
        if t+1>=len(self.offsets):
            return array([], dtype=int)
        return self.I[self.offsets[t]:self.offsets[t+1]]

def load_multiple_AER(filename, check_sorted = False, relative_time = False, directory = '.'):
    f=open(filename,'rb')
    line = f.readline()
    done = False
    res = []
    line = line.strip('\n')
    while not line == '':
        res.append(load_AER(os.path.join(directory, line), check_sorted = check_sorted, relative_time = relative_time))
        line = f.readline()
    f.close()
    return res

def load_AER(filename, check_sorted = False, relative_time = False):
    '''
    Loads AER data files for use in Brian.
    Returns a list containing tuples with a vector of addresses and a vector of timestamps (ints, unit is usually usecond).

    It can load any kind of .dat, or .aedat files.
    For index files (that point to multiple .(ae)dat files) it will return a list containing tuples as for single files.
    
    Keyword Arguments:
    If check_sorted is True, checks if timestamps are sorted,
    and sort them if necessary.
    If relative_time is True, it will set the first spike time to zero and all others relatively to that precise time (avoid negative timestamps, is definitely a good idea).
    
    Hence to use those data files in Brian, one should do:
    
    addr, timestamp =  load_AER(filename, relative_time = True)
    G = AERSpikeGeneratorGroup((addr, timestamps))
    '''
    _,ext = filename.split('.')
    ext = ext.strip('\n')
    filename = filename.strip('\n')
    directory = os.path.dirname(filename)
    if ext == 'aeidx':
        #AER data points to different AER files
        return load_multiple_AER(filename, check_sorted = check_sorted, relative_time = relative_time, directory = directory)
    elif not (ext == 'dat' or ext == 'aedat'):
        raise ValueError('Wrong extension for AER data, should be dat, or aedat, it was '+ext)

    
    # This is inspired by the following Matlab script:
    # http://jaer.svn.sourceforge.net/viewvc/jaer/trunk/host/matlab/loadaerdat.m?revision=2001&content-type=text%2Fplain
    f=open(filename,'rb')
    version=1 # default (if not found in the file)
    
    # Skip header and look for version number
    line = f.readline()
    while line[0] == '#':
        if line[:9] == "#!AER-DAT":
            version = int(float(line[9:-1]))
        line = f.readline()
    line += f.read()
    f.close()
    
    if version==1:
        print 'Loading version 1 file '+filename
        '''
        Format is: sequence of (addr = 2 bytes,timestamp = 4 bytes)
        Number format is big endian ('>')
        '''
        ## This commented paragraph is the non-vectorized version
        #nevents=len(line)/6
        #for n in range(nevents):
        #    events.append(unpack('>HI',line[n*6:(n+1)*6])) # address,timestamp
        x=fromstring(line,dtype=int16) # or uint16?
        x=x.reshape((len(x)/3,3))
        addr=x[:,0].newbyteorder('>')
        timestamp=x[:,1:].copy()
        timestamp.dtype=int32
        timestamp=timestamp.newbyteorder('>').flatten()
    else: # version==2
        print 'Loading version 2 file '+filename
        '''
        Format is: sequence of (addr = 4 bytes,timestamp = 4 bytes)
        Number format is big endian ('>')
        '''
        ## This commented paragraph is the non-vectorized version
        #nevents=len(line)/8
        #for n in range(nevents):
        #    events.append(unpack('>II',line[n*8:(n+1)*8])) # address,timestamp
        x = fromstring(line, dtype=int32).newbyteorder('>')
        addr = x[::2]
        timestamp = x[1::2]

    if check_sorted: # Sorts the events if necessary
        if any(diff(timestamp)<0): # not sorted
            ind = argsort(timestamp)
            addr,timestamp = addr[ind],timestamp[ind]
    if (timestamp<0).all():
        print 'Negative timestamps'
    
    if relative_time:
        t0 = min(timestamp)
        timestamp -= t0
    
    return addr,timestamp

def extract_DVS_event(addr):
    '''
    Extracts retina event from an address or a vector of addresses.
    
    Chip: Digital Vision Sensor (DVS)
    http://siliconretina.ini.uzh.ch/wiki/index.php
    
    Returns: x, y, polarity (ON/OFF: 1/-1)
    '''
    retina_size=128
    xmask = 0xfE # x are 7 bits (64 cols) ranging from bit 1-8
    ymask = 0x7f00 # y are also 7 bits
    xshift=1 # bits to shift x to right
    yshift=8 # bits to shift y to right
    polmask=1 # polarity bit is LSB

    x=retina_size-1-((addr & xmask) >> xshift)
    y=(addr & ymask) >> yshift
    pol=1-2*(addr & polmask) # 1 for ON, -1 for OFF
    return x,y,pol

if __name__=='__main__':
    path=r'C:\Users\Romain\Desktop\jaerSampleData\DVS128'
    filename=r'\Tmpdiff128-2006-02-03T14-39-45-0800-0 tobi eye.dat'

    addr,timestamp=load_AER(path+filename)
