# TODO: update all of this with the new interface/buffering mechanism
# TODO: decide on a good interface for online sounds, that is general

from brian import *
from numpy import *
import numpy
import wave
import array as pyarray
import time
try:
    import pygame
    have_pygame = True
except ImportError:
    have_pygame = False
try:
    from scikits.samplerate import resample
    have_scikits_samplerate = True
except (ImportError, ValueError):
    have_scikits_samplerate = False
from bufferable import Bufferable
from sounds import Sound, BaseSound

__all__ = ['OnlineSound',
                'OnlineWhiteNoise', 'OnlineWhiteNoiseBuffered',
                'OnlineWhiteNoiseShifted',
           ]

class OnlineSound(BaseSound):
    def __init__(self): 
        pass
        
    def update(self):
        pass
    
class OnlineWhiteNoise(OnlineSound):
    '''
    Noise generator which produces one sample at a time online
    input parameters are the mean mu and the variance sigma
    default mu=0, sigma=1
    '''
    def __init__(self,mu=None,sigma=None,tomux=1): 
        if mu==None:
            self.mu=0
        if sigma==None:
            self.sigma=1
        self.tomux=tomux

        
    def update(self):
        return (self.mu+sqrt(self.sigma)*randn(1))*self.tomux

class OnlineWhiteNoiseBuffered(OnlineSound):
    def __init__(self,samplerate,mu,sigma,max_abs_itd): 
        self.samplerate=samplerate
        self.length_buffer=int(max_abs_itd * self.samplerate)
        
        self.mu=mu
        self.sigma=sigma
        self.buffer=[0]*(2*self.length_buffer+1)
        
    def update(self):
        self.buffer.pop()
        self.buffer.insert(0,self.mu+self.sigma*randn(1))
        return self.buffer[self.length_buffer]
    
class OnlineWhiteNoiseShifted(OnlineSound):
    def __init__(self,samplerate,online_white_noise_buffered,shift=lambda:randn(1)*ms,time_interval=-1*ms): 
        #self.shift_applied=[] 
        self.samplerate=samplerate
        self.interval_in_sample= int(time_interval *self.samplerate)
        #print self.interval_in_sample 
        self.count=0
        self.shift=shift
        self.length_buffer=online_white_noise_buffered.length_buffer
        self.shift_in_sample=int(shift()* self.samplerate)
        if abs(self.shift_in_sample) > self.length_buffer:
            self.shift_in_sample=sign(self.shift_in_sample)*self.length_buffer
        self.ITDused=[]
        self.ITDused.append(self.shift_in_sample/self.samplerate)
        self.reference=online_white_noise_buffered
        
        
    def update(self):
        if self.count ==self.interval_in_sample:
            self.shift_in_sample=int(self.shift()* self.samplerate)
            if abs(self.shift_in_sample) > self.length_buffer:
                self.shift_in_sample=sign(self.shift_in_sample)*self.length_buffer
            self.ITDused.append(self.shift_in_sample/self.samplerate)
            #print self.shift()
            self.count=0
            
        self.count=self.count+1   
#        print self.length_buffer
#        print self.shift_in_sample
#        print self.length_buffer+1+self.shift_in_sample
        return self.reference.buffer[self.length_buffer+self.shift_in_sample]
