from brian import *

c = Clock(dt=1*ms, t=0.5*ms)

@network_operation(clock=c)
def f():
    print c.t
    if c.t>5*ms:
        c.t -= 0.01*ms
    
run(10*ms)

net = MagicNetwork()
net.prepare()
print net._update_schedule
