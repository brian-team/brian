print "Imports ok"
from cspikequeue import *
import numpy as np


queue = SpikeQueue(4, 10)
print "Queue initialized"
queue.expand()
print "Queue expanded"

for k in range(10):
    print queue.currenttime
    x = queue.peek()
    print queue.next()
