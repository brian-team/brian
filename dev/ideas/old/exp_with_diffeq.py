from brian import *
if False:
    tau=1*second
    eqs='''
    dy/dt = v/tau : 1
    dv/dt = -y/tau : 1
    dx/dt = exp(y)/tau : 1
    dw/dt = z/tau : 1 # w should be the same as x
    dz/dt = v*z/tau : 1
    '''
    G=NeuronGroup(1, eqs)
    M=MultiStateMonitor(G, record=True, when='start')
    G.y=0
    G.v=1
    G.x=0
    G.w=G.x
    G.z=exp(G.y)
    run(1*second)
    M.plot()
    legend()
    show()
if True:
    tau=1*second
    eqs='''
    dy/dt=exp(y)/tau:1
    dx/dt=u/tau:1
    du/dt=u*u/tau:1
    '''
    G=NeuronGroup(1, eqs)
    M=MultiStateMonitor(G, record=True, when='start')
    G.y=1
    G.x=G.y
    G.u=exp(G.y)
    run(exp(-1)*0.99*second)
    z=-log(-M.times+exp(-1))
    subplot(211)
    plot(M.times, M['x', 0], label='dx/dt=u, du/dt=u*u')
    plot(M.times, M['y', 0], label='dy/dt=exp(y)')
    plot(M.times, z, label='analytic soln') # analytic solution
    legend(loc='upper left')
    subplot(212)
    plot(M.times, abs(M['x', 0]-z), label='abs err in x')
    plot(M.times, abs(M['y', 0]-z), label='abs err in y')
    legend(loc='upper left')
    print 'max err in x', max(abs(M['x', 0]-z))
    print 'max err in y', max(abs(M['y', 0]-z))
    show()
