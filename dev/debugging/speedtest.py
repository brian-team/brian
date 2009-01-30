from brian import *
from time import time

P=NeuronGroup(1,model='x:1',threshold=1,reset=0)
C=Connection(P,P,'x')
run(1*ms)
t1=time()
run(10*second)
t2=time()
print t2-t1
