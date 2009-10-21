from brian import *
from brian.library.ionic_currents import *
from IPython.Shell import IPShellEmbed
ipshell = IPShellEmbed()

N = 100
duration = 1*second

El=10.6*mV
EK=-12*mV
ENa=120*mV
eqs=MembraneEquation(1*uF)+leak_current(.3*msiemens,El)
eqs+=K_current_HH(36*msiemens,EK)+Na_current_HH(120*msiemens,ENa)
eqs+=Current('I:amp')
neuron=NeuronGroup(N, eqs, implicit=True, freeze=True)
trace=RecentStateMonitor(neuron, 'vm', record=[0,1,2], duration=50*ms)
neuron.I = 10*uA*rand(N)

ion()
trace.plot(refresh=1*ms)
net = MagicNetwork()

while defaultclock.t<duration:
    if hasattr(net, 'stopped') and net.stopped:
        break
    try:
        print 'Starting at time', defaultclock.t
        net.run(duration-defaultclock.t)
    except KeyboardInterrupt:
        print 'Interrupted at time', defaultclock.t
        ipshell(local_ns=locals(), global_ns=globals())
print 'Finished at time', defaultclock.t
ioff()
clf()
trace.plot()
show()
