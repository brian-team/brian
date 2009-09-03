import brian_no_units
from brian import *
reinit()

#RS
a=0.02
b=0.2
c=-65
d=8


#---------------------------------------------------------------------------------

defaultclock.dt=0.001

n=2 #number of neurons in the network
r=20 #strength of external input



eqs=Equations(''' dv/dt= 0.04*v**2 + 5*v + 140 - u + I: 1
du/dt=a*(b*v-u):1
I:1
''')


PG=PoissonGroup(n,rates=2*Hz)

G = NeuronGroup (n, model=eqs, reset="v=c;u=u+d",threshold='v>=30')



@network_operation(when='after_groups')
def Icurrent():
    G.I=10


C2=IdentityConnection(PG,G,'I',weight=r)

G.v=-65
G.u=b*G.v
G.I=10

Sp=SpikeMonitor(G)
M=MultiStateMonitor(G,record=True)



net=Network(Icurrent,PG,G,C2,Sp,M)


net.run(140)



figure(1)
raster_plot(Sp)


figure(2)
subplot(4,1,1)
plot(M['v'].times,M['v'][0])
ylabel('v (dimensionless)')
subplot(4,1,2)
plot(M['I'].times,M['I'][0])
ylabel('I (dimensionless)')
subplot(4,1,3)
plot(M['u'].times,M['u'][0])
xlabel('Time/seconds')
ylabel('u (dimensionless)')


figure(3)
subplot(4,1,1)
plot(M['v'].times,M['v'][1])
ylabel('v (dimensionless)')
subplot(4,1,2)
plot(M['I'].times,M['I'][1])
ylabel('I (dimensionless)')
subplot(4,1,3)
plot(M['u'].times,M['u'][1])
xlabel('Time/seconds')
ylabel('u (dimensionless)')


show()