# From Ariel Zylberberg
from brian import *

vl = -70 * mvolt
vth = -50 * mvolt
vreset = -55 * mvolt
tauAMPA = 2 * ms
tauNMDA = 100 * ms
tauGABA = 10 * ms
a = 0.062 * mvolt ** -1
b = 3.57 * 1
Mg = 1
ve = 0 * mvolt
vi = -70 * mvolt

eqs = '''
    dv/dt = (-gl*(v-vl)-Isyn)*(1./cm) : volt
    Isyn  = IAMPA+INMDA+IGABA : amp
    IAMPA = (gextAMPA*sextAMPA+grecAMPA*srecAMPA)*(v-ve) : amp
    INMDA = gNMDA*sNMDA*(v-ve)/(1+Mg*exp(-a*v)/b) : amp
    IGABA = gGABA*sGABA*(v-vi) : amp
    dsextAMPA/dt = -sextAMPA/tauAMPA : 1
    dsrecAMPA/dt = -srecAMPA/tauAMPA : 1
    dsNMDA/dt    = -sNMDA/tauNMDA : 1 
    dsGABA/dt    = -sGABA/tauGABA : 1
    dss/dt = (0.63-ss)/tauNMDA : 1
    cm : nfarad
    gl : nsiemens
    gextAMPA : nsiemens
    grecAMPA : nsiemens
    gNMDA : nsiemens
    gGABA : nsiemens
    '''

N = 1000
Ne = 800
Ni = 200

Ge = NeuronGroup(Ne, model=eqs, threshold=vth, reset=vreset, refractory=2 * msecond)
Gi = NeuronGroup(Ni, model=eqs, threshold=vth, reset=vreset, refractory=1 * msecond)

Ge.cm = 0.5 * nfarad
Ge.gl = 25 * nsiemens
Ge.gextAMPA = 2.08 * nsiemens
Ge.grecAMPA = 0.104 * nsiemens
Ge.gNMDA = 0.327 * nsiemens
Ge.gGABA = 1.25 * nsiemens

Gi.cm = 0.2 * nfarad
Gi.gl = 20 * nsiemens
Gi.gextAMPA = 1.62 * nsiemens
Gi.grecAMPA = 0.081 * nsiemens
Gi.gNMDA = 0.258 * nsiemens
Gi.gGABA = 0.973 * nsiemens

#init
Ge.ss = 0
Ge.sNMDA = 0
Ge.sGABA = 0
Ge.sextAMPA = 0
Ge.srecAMPA = 0
Ge.v = -60 * mV

Gi.ss = 0
Gi.sNMDA = 0
Gi.sGABA = 0
Gi.sextAMPA = 0
Gi.srecAMPA = 0
Gi.v = -60 * mV

#recurrent connections
def reset_fun(P, spikes):
    P.v_[spikes] = vreset
    P.ss_[spikes] = 0

C = []
init_list = [(Ge, Ge, 'srecAMPA', None),
           (Ge, Gi, 'srecAMPA', None),
           (Ge, Ge, 'sNMDA', 'ss')]
for source, target, variable, modulation in init_list:
    C.append(Connection(source, target, variable, modulation=modulation))
    C[-1].connect_full(source, target, weight=1)

C1 = Connection(Ge, Ge, 'srecAMPA')
C1.connect_full(Ge, Ge, weight=1)

C2 = Connection(Ge, Gi, 'srecAMPA')
C2.connect_full(Ge, Gi, weight=1)

C3 = Connection(Ge, Ge, 'sNMDA', modulation='ss')
C3.connect_full(Ge, Ge, weight=1)

C4 = Connection(Ge, Gi, 'sNMDA', modulation='ss')
C4.connect_full(Ge, Gi, weight=1)

C5 = Connection(Gi, Ge, 'sGABA')
C5.connect_full(Gi, Ge, weight=1)

C6 = Connection(Gi, Gi, 'sGABA')
C6.connect_full(Gi, Gi, weight=1)

#external connections
Pe = PoissonGroup(Ne, 2400 * Hz)
Ce = Connection(Pe, Ge, 'sextAMPA')
Ce.connect_one_to_one(Pe, Ge, weight=1)

Pi = PoissonGroup(Ni, 2400 * Hz)
Ci = Connection(Pi, Gi, 'sextAMPA')
Ci.connect_one_to_one(Pi, Gi, weight=1)

#record
Me = PopulationRateMonitor(Ge, bin=25 * ms)
Mi = PopulationRateMonitor(Gi, bin=25 * ms)

#run
run(10.0 * second)

#plot
plot(Me.times / ms, Me.rate / Hz)
plot(Mi.times / ms, Mi.rate / Hz)
show()


