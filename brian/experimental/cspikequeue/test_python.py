print "Imports ok"
import sys
if not '.' in sys.path: sys.path.append('.')
from cspikequeue import *
import numpy as np


queue = SpikeQueue(4, 10)
sys.stdout.flush()
print "Queue initialized"
sys.stdout.flush()
print 'currenttime', queue.currenttime
print 'n', queue.n
print 'X', queue.X
#queue.minimal()
q = queue.peek()
print "Queue peeked!"


# #print queue
# #queue.expand()
# print "Queue expanded"

for k in range(10):
    print queue.currenttime
    x = queue.peek()
    print x
    print queue.next()


