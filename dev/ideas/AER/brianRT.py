'''
Realtime Brian

Pauses are inserted so that Brian runs always between real time and
about 50 ms ahead of it (probably a bit more).
Principle: every 50 ms, we check whether Brian is ahead by more than
100 ms, and if so we insert a sleep() to synchronize.

NB: of course Brian needs to be faster than real time!
(so use a large timestep)

Other features of this real time stuff:
* set the offset of the real time clock (reference time)
'''
from brian import *
from time import time,sleep

defaultclock.dt=0.5*ms

G=NeuronGroup(1000,model='dv/dt=-v/(10*ms):1')

first_time=True
@network_operation(EventClock(dt=50*ms))
def catch_up(cl):
    global start_time,first_time
    # First time: synchronize Brian and real time
    if first_time:
        start_time=time()
        first_time=False
    real_time=time()-start_time
    print cl.t,real_time
    if cl._t>real_time:
        sleep(cl._t-real_time)
    
run(10*second)
