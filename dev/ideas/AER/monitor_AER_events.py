"""
Monitor AER events with real-time Brian.

I think it works (although not sure) but it's really too slow.
It needs to be vectorised.
"""
#import brian_no_units
from brian import *
import socket
from struct import *
from time import time,sleep

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

# Open the UDP socket
UDP_IP="localhost"
UDP_PORT=8991
sock = socket.socket( socket.AF_INET, # Internet
                      socket.SOCK_DGRAM ) # UDP
sock.bind( (UDP_IP,UDP_PORT) )
sock.setblocking(0) # non-blocking

defaultclock.dt=1*ms # so that it's fast enough

# This is real-time Brian!
first_time=True
@network_operation(EventClock(dt=50*ms))
def catch_up(cl):
    global start_time,first_time
    # First time: synchronize Brian and real time
    if first_time:
        start_time=time()
        first_time=False
    real_time=time()-start_time
    #print cl.t,real_time
    if cl._t>real_time:
        sleep(cl._t-real_time)
    
retina=SpikeGeneratorGroup(128,[])
    
# Listen to AER
first_event=True
offset=0.
latency=200*ms
@network_operation(EventClock(dt=20*ms))
def listen():
    global first_event,offset
    try:
        while True:
            data= sock.recvfrom( 63000 )[0]
            packet_number=unpack('>I',data[:4]) # not very useful
            events=[unpack('>II',data[(8*n+4):(8*n+12)]) for n in range(len(data)/8-1)]
            if (len(events)>0):
                if first_event: # Synchronize event timestamps and clock
                    first_event=False
                    t0=time()-start_time
                    ts0=events[0][1]*1e-6
                    offset=t0-ts0
                spikelist=[(pixel_to_neuron(*extract_retina_event(event)),(t*1e-6+offset)*second+latency)\
                           for (event,t) in events[:1]] #only first event because it's so slow!
                retina.spiketimes+=spikelist
    except: # no more events (can we do this in a cleaner way?)
        pass

M=SpikeMonitor(retina)

run(5*second)

sock.close()

raster_plot(M)
show()
