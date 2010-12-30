from brian import *

c1 = Clock(dt=2*ms, order=1)
c2 = Clock(dt=3*ms, order=2)
c3 = Clock(dt=5*ms, order=0)

@network_operation(clock=c1)
def f1():
    print '1:', c1.t
    
@network_operation(clock=c2)
def f2():
    print '2:', c2.t

@network_operation(clock=c3)
def f3():
    print '3:', c3.t

net = MagicNetwork()

net.run(20*ms)

print c1.t, c2.t, c3.t
