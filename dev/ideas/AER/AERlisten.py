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
from brian.experimental.neuromorphic import *

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
    #events=[unpack('>II',data[(8*n+4):(8*n+12)]) for n in range(len(data)/8-1)]
    events=fromstring(data[4:],int32).newbyteorder('>')
    # something's wrong with the times
    
    if (len(events)>0):
        addr=events[::2]
        timestamp=events[1::2]
        if first_time: # Synchronize event timestamps and clock
            first_time=False
            t0=time()
            #ts0=events[0][1]
            ts0=timestamp[0]
        #ts=events[0][1]-ts0
        ts=timestamp[0]-ts0
        # Show first timestamp, clock time, first event
        print ts*1e-6,time()-t0,extract_DVS_event(addr[0])#,timestamp[0] #events[0][0])
    
sock.close()
