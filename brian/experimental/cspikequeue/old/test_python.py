print "Imports ok"
import sys
if not '.' in sys.path: sys.path.append('.')
from cspikequeue import *
#from cspikequeue.cspikequeue_pythonmodule import CSpikeQueueWrapper
import numpy as np

if True:
    queue = SpikeQueue(20, 10)
    #print queue
    print "time", queue.currenttime
    sys.stdout.flush()
    q = queue.peek()

    spikes = np.array([0, 1, 2, 3], dtype = int)
    delays = np.array([2, 3, 4, 5], dtype = int)

    queue.insert(spikes, delays)

    queue.expand(100)
    queue.expand(100)
    queue.expand(100)
    queue.expand(100)
    queue.expand(100)
    queue.expand(100)

    queue.insert(spikes+1, delays)

    print queue
    # print "Queue expanded"

    for k in range(10):
        print "time", queue.currenttime
        print "current content", queue.peek()
        queue.next()
