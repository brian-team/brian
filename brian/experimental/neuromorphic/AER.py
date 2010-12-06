"""
Module to deal with the AER (Address Event Representation) format.
"""
from struct import *

__all__=['load_AER','extract_DVS_event']

def load_AER(filename):
    '''
    Loads an AER .dat file and returns a list of events
    as (address, timestamp) (ints)
    
    timestamp is (probably) in microseconds
    
    TODO: return arrays
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

def extract_DVS_event(addr):
    '''
    Extracts retina event from an address
    
    Chip: Digital Vision Sensor (DVS)
    http://siliconretina.ini.uzh.ch/wiki/index.php
    
    Returns: x, y, polarity (ON/OFF: 1/-1)
    
    TODO:
    * vectorise
    * maybe this should in a "chip" module?
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

