from brian import *

class FutureSpikeContainer(object):
    def __init__(self, nsteps):
        self.nsteps = nsteps
        self.maxspikes = 2
        self.ind = zeros((2, nsteps), dtype=int)
        self.nspikes = zeros(nsteps, dtype=int)
        self.cursor = 0
    def insert(self, spikes, delays):
        t = (self.timecursor+delays)%self.nsteps
        I = argsort(t)
        t = t[I]
        spikes = spikes[I]
        # TODO: finish this - use scheduled_events.py which basically does the
        # same thing
        
        # assume for the moment that no overflows happen
    def getspikes(self):
        spikes = self.ind[0:self.nspikes[self.cursor], self.cursor]
        self.nspikes[timecursor] = 0
        self.timecursor += 1
        if self.timecursor==self.nsteps:
            self.timecursor = 0
        return spikes

if __name__=='__pass__':
    pass
