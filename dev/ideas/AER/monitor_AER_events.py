"""
Monitor AER events with real-time Brian.

I think it works (although not sure) but it's really too slow.
It needs to be vectorised.

Problems with negative timestamps
"""
#import brian_no_units
from brian import *
import socket
from struct import *
from time import time,sleep
from brian.experimental.neuromorphic import *

# Open the UDP socket
UDP_IP="localhost"
UDP_PORT=8991
sock = socket.socket( socket.AF_INET, # Internet
                      socket.SOCK_DGRAM ) # UDP
sock.bind( (UDP_IP,UDP_PORT) )
sock.setblocking(0) # non-blocking

defaultclock.dt=1*ms # so that it's fast enough

R=RealtimeController(dt=100*ms)
   
#retina=SpikeGeneratorGroup(128,[])
allspikes=[]

# Listen to AER
first_event=True
offset=0.
latency=200*ms
@network_operation(EventClock(dt=20*ms))
def listen():
    global first_event,offset,allspikes
    try:
        while True:
            data= sock.recvfrom( 63000 )[0]
            #packet_number=unpack('>I',data[:4]) # not very useful
            #events=[unpack('>II',data[(8*n+4):(8*n+12)]) for n in range(len(data)/8-1)]
            events=fromstring(data[4:],int32).newbyteorder('>')
            if (len(events)>0):
                events=events[:32] # a small selection
                addr=events[::2]
                timestamp=events[1::2]
                if first_event: # Synchronize event timestamps and clock
                    first_event=False
                    t0=time()-start_time
                    ts0=timestamp[0]*1e-6
                    offset=t0-ts0
                _,addr,_=extract_DVS_event(addr)
                #spiketimes=array((addr,timestamp*1e-6+offset+float(latency))).T
                #spiketimes=retina.gather(spiketimes,defaultclock.dt)
                allspikes+=zip(addr,timestamp*1e-6+offset+float(latency))
    except: # no more events (can we do this in a cleaner way?)
        pass

#M=SpikeMonitor(retina)

start_time=time()
run(5*second)

sock.close()

i,t=zip(*allspikes)
plot(t,i,'.')
#raster_plot(M)
show()
