from brian import *


class RandomEventClock(EventClock):
    def __init__(self, t=0*ms, dt=1*second, randomfunc=None):
        EventClock.__init__(self, t=t, dt=dt)
        if randomfunc is None:
            randomfunc=rand
        self.randomfunc=randomfunc

    def tick(self):
        self._t+=self.randomfunc()*self._dt

G=NeuronGroup(1, 'V:1')
M=StateMonitor(G, 'V', record=True)

@network_operation(clock=RandomEventClock())
def f():
    G.V+=1

run(5*second)

M.plot()
show()
