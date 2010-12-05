"""
Real-time Brian

See BEP-20-AER interface
"""
from brian.network import NetworkOperation
from brian.clock import EventClock
from brian.stdunits import ms
from time import sleep,time

__all__=['RealtimeController']

class RealtimeController(NetworkOperation):
    '''
    Pauses are inserted so that Brian runs always between real time and
    about 50 ms ahead of it (probably a bit more).
    Principle: every 50 ms, so we insert a sleep() to synchronize.
    
    Typical use::
      R=RealtimeController()
      run(1*second)
      R.reinit() # will resynchronise real time at next run
      run(2*second)
    '''
    def __init__(self,dt=50*ms,verbose=False):
        '''
        dt is the resynchronisation period
        '''
        self.clock = EventClock(dt=dt) # this could be a global preference
        self.when = 'end'
        self.first_time=True
        self.verbose=verbose # for debugging

    def __call__(self):
        # First time: synchronise Brian and real time
        if self.first_time: # stores the start time
            self.synchronise()
            self.first_time=False
        real_time=time()+self.offset
        if self.verbose:
            print self.clock.t,real_time
        if self.clock._t>real_time: # synchronize real time and clock
            sleep(self.clock._t-real_time)
            
    def synchronise(self):
        '''
        Sets the offset to the real clock so that Brian and real time are
        synchronised.
        '''
        self.offset=self.clock._t-time()
        
    def reinit(self):
        '''
        The clock synchronise with real time at the next call.
        '''
        self.first_time=True

if __name__=='__main__':
    from brian import *
    
    R=RealtimeController(dt=100*ms,verbose=True)
    run(3*second)
    sleep(1*second)
    R.reinit() # try not comment it!
    run(2*second)
    