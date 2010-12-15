"""
Module to deal with the AER (Address Event Representation) format.
"""
#from struct import *
from numpy import *

__all__=['load_AER','extract_DVS_event']

def load_AER(filename):
    '''
    Loads an AER .dat file and returns
    a vector of addresses and a vector of timestamps (ints)
    
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
    
    if version==1:
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
        #nevents=len(line)/8
        #for n in range(nevents):
        #    events.append(unpack('>II',line[n*8:(n+1)*8])) # address,timestamp
        x=fromstring(line,dtype=int32).newbyteorder('>')
        addr=x[:,0]
        timestamp=x[:,1]

    return addr,timestamp

def extract_DVS_event(addr):
    '''
    Extracts retina event from an address
    
    Chip: Digital Vision Sensor (DVS)
    http://siliconretina.ini.uzh.ch/wiki/index.php
    
    Returns: x, y, polarity (ON/OFF: 1/-1)
    
    TODO:
    * maybe this should in a "chip" module?
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
