from brian import *

@network_operation
def f():
    print "yo"

run(1*ms)
