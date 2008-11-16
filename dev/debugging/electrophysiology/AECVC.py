# AEC experiment (voltage-clamp)
from brian import *
from brian.library.electrophysiology import *

myclock=Clock(dt=.1*ms)
clock_rec=Clock(dt=.1*ms)

#log_level_debug()

taum=20*ms
gl=20*nS
Cm=taum*gl
Re=50*Mohm
Ce=0.1*ms/Re

eqs=Equations('''
dvm/dt=(-gl*vm+i_inj)/Cm : volt
I:amp
''')
eqs+=electrode(.6*Re,Ce)
eqs+=current_clamp(vm='v_el',i_inj='i_cmd',i_cmd='I',Re=.4*Re,Ce=Ce)
setup=NeuronGroup(1,model=eqs,clock=myclock)
board=VC_AEC(setup,'v_rec','I',gain=300*nS,gain2=20*nS/ms,clock=clock_rec)
recording=StateMonitor(board,'I',record=True,clock=myclock)
soma=StateMonitor(setup,'vm',record=True,clock=myclock)

run(50*ms)
board.start_injection()
run(1*second)
board.stop_injection()
run(100*ms)
board.estimate()
print 'Re=',sum(board.Ke)*ohm
board.switch_on()
run(50*ms)
board.command(10*mV)
run(200*ms)
board.command(0*mV)
run(150*ms)
board.switch_off()
figure()
plot(recording.times/nA,recording[0]/nA,'b')
figure()
plot(soma.times/ms,soma[0]/mV,'r')
show()

