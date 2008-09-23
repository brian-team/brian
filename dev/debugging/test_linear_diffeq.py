from brian import *

tau = 20*ms

# one const
eg_a = {'eqs' : '''
                dx/dt = 1/tau : 1
                dy/dt = -x/tau : 1
                ''',
        'init_x' : 1,
        'init_y' : 0,
        # analytic solution from Mathematica
        'sol_x' : lambda t: 1. + 50.*t,
        'sol_y' : lambda t: -50.*(t + 25.*t**2)
        }
        #dx/dt = 1/tau : 1
        #dy/dt = -x/tau : 1
        #x-error at end 1.98063787593e-013
        #y-error at end 0.0012487500008 (num euler steps=10)
        #y-error at end 0.000124875000608 (num euler steps=100, the default value)
        #y-error at end 1.2487501337e-005 (num euler steps=1000)

# both const
eg_b = {'eqs' : '''
                dx/dt = 1/tau : 1
                dy/dt = 2/tau : 1
                ''',
        'init_x' : 1,
        'init_y' : 0,
        # analytic solution
        'sol_x' : lambda t: 1. + 50.*t,
        'sol_y' : lambda t: 100.*t
        }
        #dx/dt = 1/tau : 1
        #dy/dt = 2/tau : 1
        #
        #x-error at end 1.98063787593e-013
        #y-error at end 3.51718654201e-013

# neither const, this one uses the expm analytic method
eg_c = {'eqs' : '''
                dx/dt = y/tau : 1
                dy/dt = -x/tau : 1
                ''',
        'init_x' : 1,
        'init_y' : 0,
        # analytic solution
        'sol_x' : lambda t: cos(50.*t),
        'sol_y' : lambda t: -1.*sin(50.*t)
        }
        #dx/dt = y/tau : 1
        #dy/dt = -x/tau : 1
        #x-error at end 8.77631300966e-014
        #y-error at end 2.3758772727e-014


eg = eg_a

eqs = eg['eqs']

print '\n'.join(l.strip() for l in eqs.split('\n'))

G = NeuronGroup(1,eqs)

Mx = StateMonitor(G, 'x', record=True, when='start')
My = StateMonitor(G, 'y', record=True, when='start')

G.x = eg['init_x']
G.y = eg['init_y']

run(100*ms)

x = eg['sol_x']
y = eg['sol_y']

print 'x-error at end', abs(float(Mx[0][-1])-x(Mx.times_[-1]))
print 'y-error at end', abs(float(My[0][-1])-y(My.times_[-1]))

plot(Mx.times/ms, Mx[0])
plot(Mx.times/ms, x(Mx.times_),'--')
plot(My.times/ms, My[0])
plot(My.times/ms, y(My.times_),'--')

show()