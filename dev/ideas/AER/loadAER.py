"""
Loads an AER .dat file and plays it in Brian

Version:
1 -    addr is 2 bytes, timestamp is 4 bytes (default)
2 -    addr is 4 bytes, timestamp is 4 bytes
"""
from brian import *
from struct import *

def load_AER(filename):
    '''
    Loads an AER .dat file and returns a list of events
    as (address, timestamp) (ints)
    
    timestamp is (probably) in microseconds
    '''
    f=open(filename,'rb')
    version=1 # default
    
    # Skip header and look for version number
    line=f.readline()
    while line[0]=='#':
        if line[:9]=="#!AER-DAT":
            version=int(float(line[9:-1]))
        line=f.readline()
    line+=f.read()
    f.close()
    
    events=[]
    if version==1:
        nevents=len(line)/6
        for n in range(nevents):
            events.append(unpack('>HI',line[n*6:(n+1)*6])) # address,timestamp
    else: # version==2
        nevents=len(line)/8
        for n in range(nevents):
            events.append(unpack('>II',line[n*8:(n+1)*8])) # address,timestamp

    return events

def extract_retina_event(addr):
    '''
    Extract retina event from an address
    
    Returns: x, y, polarity (ON/OFF: 1/-1)
    TODO: vectorise
    '''
    retina_size=128
    xmask = 0xfE # x are 7 bits (64 cols) ranging from bit 1-8
    ymask = 0x7f00 # y are also 7 bits
    xshift=1 # bits to shift x to right
    yshift=8 # bits to shift y to right
    polmask=1 # polarity bit is LSB

    if addr<0:
        print "negative address!"

    x=retina_size-1-((addr & xmask) >> xshift)
    y=(addr & ymask) >> yshift
    pol=1-2*(addr & polmask) # 1 for ON, -1 for OFF
    return x,y,pol

def pixel_to_neuron(x,y,pol):
    return y+0*x # let's just have 128 neurons, one per row for now

path=r'C:\Users\Romain\Desktop\jaerSampleData\DVS128'
filename=r'\Tmpdiff128-2006-02-03T14-39-45-0800-0 tobi eye.dat'

events=load_AER(path+filename)
spikes=[(pixel_to_neuron(*extract_retina_event(addr)),t*1e-6*second) for (addr,t) in events]
# assuming microsecs
print spikes[-1]

P=SpikeGeneratorGroup(128,spikes)
M=SpikeMonitor(P)

print "starting"
run(1*second)

raster_plot(M)
show()
