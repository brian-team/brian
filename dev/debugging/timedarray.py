from brian import *
import time

#x = TimedArray([1,2,3,4,5], dt=1*ms)
x = TimedArray(array([[1,2,3,4,5],[6,7,8,9,10]]).T, dt=1*ms)
#print x([1.5*ms, 2.5*ms, 3.5*ms])
print x([1.5*ms, 2.5*ms])
exit()

defaultclock.dt = 1*ms
#x = array([t*(arange(10)+randn(10)) for t in arange(1000)/1000.])
#x = TimedArray(x, dt=.5*ms)
#x = x[100:200]
x = array([zeros(10),arange(10)/2.,zeros(10)])
x = TimedArray(x, times=array([100*ms, 200*ms, 500*ms]))
#G = NeuronGroup(10, '''dV/dt=(-V+x(t))/(10*ms):1''', threshold=1, reset=0)
G = NeuronGroup(10, '''dV/dt=(-V+I)/(10*ms):1
                       I : 1''', threshold=1, reset=0)
#set_group_var_by_array(G, 'I', x)
#G.set_var_by_array('I', x)
G.I = x
print G.contained_objects[0].clock.dt
print G.clock.dt
M = MultiStateMonitor(G, record=True)
start = time.time()
run(1*second)
print time.time()-start
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