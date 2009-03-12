from brian import *
defaultclock.dt = 1*ms
x = array([t*(arange(10)+randn(10)) for t in arange(1000)/1000.])
#x = TimedArray(x)
#G = NeuronGroup(10, '''dV/dt=(-V+x(t))/(10*ms):1''', threshold=1, reset=0)
G = NeuronGroup(10, '''dV/dt=(-V+I)/(10*ms):1
                       I : 1''', threshold=1, reset=0)
#set_group_var_by_array(G, 'I', x)
G.set_var_by_array('I', x)
M = MultiStateMonitor(G, record=True)
run(1*second)
M.plot()
legend()
show()
#    y0 = []
#    for i in range(1000):
#        #print i
#        y0.append(y(i*ms)[0])
#    plot(arange(1000)/1000., y0)
#    y[:,0].plot()
#    show()
#    z = y[1,1:5]
#    print z.shape
#    print asarray(z)
#    print z.times
#    #z.plot()
#    #show()
##    for i in range(10):
##        y[10:30,i].plot()
#    y[10:30,:].plot()
#    legend()
#    show()