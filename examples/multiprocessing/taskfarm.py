#!/usr/bin/env python
'''
Uses the :func:`~brian.tools.taskfarm.run_tasks` function to run a task on
multiple CPUs and save the results to a
:class:`~brian.tools.datamanager.DataManager` object.
'''
from brian import *
from brian.tools.datamanager import *
from brian.tools.taskfarm import *

def find_rate(k, report):
    eqs = '''
    dV/dt = (k-V)/(10*ms) : 1
    '''
    G = NeuronGroup(1000, eqs, reset=0, threshold=1)
    M = SpikeCounter(G)
    run(30*second, report=report)
    return (k, mean(M.count)/30)

if __name__=='__main__':
    N = 20
    dataman = DataManager('taskfarmexample')
    if dataman.itemcount()<N:
        M = N-dataman.itemcount()
        run_tasks(dataman, find_rate, rand(M)*19+1)
    X, Y = zip(*dataman.values())
    plot(X, Y, '.')
    xlabel('k')
    ylabel('Firing rate (Hz)')
    show()
