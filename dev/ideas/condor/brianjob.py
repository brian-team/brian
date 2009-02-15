'''
This a job to be executed by Condor
'''
from brian import *
#import sys

#arg=sys.argv[1]

from time import time

print "Hi everyone!"
print "Here is a voltage:",randint(100)*mV
