print "Imports ok"
import sys
if not '.' in sys.path: sys.path.append('.')
from cspikequeue import *
import numpy as np


queue = SpikeQueue(20, 10)
#print queue
print "time", queue.currenttime
sys.stdout.flush()
q = queue.peek()

spikes = np.array([0, 1, 2, 3], dtype = int)
delays = np.array([2, 3, 4, 5], dtype = int)

queue.insert(spikes, delays)

# #queue.expand()
# print "Queue expanded"

for k in range(10):
    print "time", queue.currenttime
    print "current content", queue.peek()
    queue.next()
