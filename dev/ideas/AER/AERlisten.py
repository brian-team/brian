"""
Listens to AER packets through UDP.

Format:
int32 sequenceNumber
int32 address0
int32 timestamp0
int32 address1
int32 timestamp2
etc
"""
from scipy import *
import socket
from struct import *
from time import time

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

UDP_IP="localhost"
UDP_PORT=8991

sock = socket.socket( socket.AF_INET, # Internet
                      socket.SOCK_DGRAM ) # UDP
sock.bind( (UDP_IP,UDP_PORT) )

first_time=True
t0=1e12
while time()-t0<5.: # run for 5 s
    data, _ = sock.recvfrom( 63000 ) # buffer size
    packet_number=unpack('>I',data[:4])
    # can we vectorize?
    # events are (address, timestamp)
    events=[unpack('>II',data[(8*n+4):(8*n+12)]) for n in range(len(data)/8-1)]
        
    if (len(events)>0):
        if first_time: # Synchronize event timestamps and clock
            first_time=False
            t0=time()
            ts0=events[0][1]
        ts=events[0][1]-ts0
        # Show first timestamp, clock time, first event
        print ts*1e-6,time()-t0,extract_retina_event(events[0][0])
    
sock.close()
