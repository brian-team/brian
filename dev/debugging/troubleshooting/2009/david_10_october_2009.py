"""
A model of working memory from Wang (2002) implemented in Brian
"""
import time

time_start=time.time()
debugg=True      # set True for debugging (checks the units)

if not debugg:
  import brian_no_units

from brian import *


# === Define parameters
# size of the network
excit_n=1600
inhib_n=400

# simulation-related parameters
simtime=1000.0*ms     # total simulation time [ms]
dt=0.02*ms       # simulation step length [ms]

# === Parameters determining model dynamics =====
# pyramidal cells
Cm_e=0.5*nF     # [nF] total capacitance
gl_e=25.0*nS    # [ns] total leak conductance
El_e=-70.0*mV   # [mV] leak reversal potential
Vth_e=-50.0*mV   # [mV] threshold potential
Vr_e=-55.0*mV   # [mV] resting potential
tr_e=2.0*ms   # [ms] refractory time

# interneuron cells
Cm_i=0.2*nF   # [nF] total capacitance
gl_i=20.0*nS   # [ns] total leak conductance
El_i=-70.0*mV    # [mV] leak reversal potential
Vth_i=-50.0*mV    # [mV] threshold potential
Vr_i=-55.0*mV    # [mV] resting potential
tr_i=1.0*ms     # [ms] refractory time

#structure
N1=240
N2=2*N1
f=0.15
wp=1.7
wm=1.0-f*(wp-1.0)/(1.0-f)

def structure(i, j):
    if(i==j): return 0
    if (i<N1):
        if (j<N1): return wp
        elif (j<N2): return wm
        else: return 1.0
    elif (i<N2):
        if (j<N1): return wm
        elif (j<N2): return wp
        else: return 1.0
    else:
        if (j<N2): return wm
        else: return 1.0

# external input
fext=2400.0*Hz   # [Hz] external input frequency (poisson train)
def stimulus(t):
  if (t<1*second): return fext
  elif (t<1.5*second): return 2440.0*Hz
  else: return fext

# AMPA receptor (APMAR)
Vs_ampa=0.0*mV     # [mV] synaptic reversial potential
t_ampa=2.0*ms     # [ms] exponential decay time constant
g_ext_e=2.1*nS     # [nS] maximum conductance from external to pyramidal cells
g_ext_i=1.62*nS    # [nS] maximum conductance from external to interneuron cells
g_ampa_i=0.04*nS
g_ampa_e=0.05*nS

# GABA receptor (GABAR)
Vs_gaba=-70.0*mV      # [mV] synaptic reversial potential
t_gaba=5.0*ms   # [ms] exponential decay time constant
g_gaba_e=1.3*nS
g_gaba_i=1.0*nS

# NMDA receptor (NMDAR)
Vs_nmda=0.0*mV     # [mV] synaptic reversial potential
ts_nmda=100.0*ms   # [ms] decay time of NMDA currents
tx_nmda=2.0*ms   # [ms] controls the rise time of NMDAR channels
alfa_nmda=0.5*kHz      # [kHz] controls the saturation properties of NMDAR channels
g_nmda_e=0.165*nS
g_nmda_i=0.13*nS


#################################

eqs_e='''
dv/dt = (-gl_e*(v-El_e)-g_ext_e*(v-Vs_ampa)*s_ext-g_gaba_e*(v-Vs_gaba)*s_gaba-g_ampa_e*(v-Vs_ampa)*s_ampa-s_tot*g_nmda_e*(v-Vs_nmda)/(1+exp(-0.062*v/(1*mV))/3.57))/Cm_e: volt
ds_ext/dt = -s_ext/(t_ampa) : 1
ds_gaba/dt = -s_gaba/(t_gaba) : 1
ds_ampa/dt = -s_ampa/(t_ampa) : 1
ds_nmda/dt = -s_nmda/(ts_nmda)+alfa_nmda*x*(1-s_nmda) : 1
dx/dt = -x/(tx_nmda) : 1
s_tot : 1
'''
eqs_i='''
dv/dt = (-gl_i*(v-El_i)-g_ext_i*(v-Vs_ampa)*s_ext-g_gaba_i*(v-Vs_gaba)*s_gaba-g_ampa_i*(v-Vs_ampa)*s_ampa-s_tot*g_nmda_i*(v-Vs_nmda)/(1+exp(-0.062*v/(1*mV))/3.57))/Cm_i: volt
ds_ext/dt = -s_ext/(t_ampa) : 1
ds_gaba/dt = -s_gaba/(t_gaba) : 1
ds_ampa/dt = -s_ampa/(t_ampa) : 1
s_tot : 1
'''

simulation_clock=Clock(dt=dt)

# Setting up the populations
print "Setting up the populations ..."


Pe=NeuronGroup(excit_n, eqs_e, threshold=Vth_e, reset=Vr_e, refractory=tr_e, order=2)
Pe.v=El_e
Pe.s_ext=0
Pe.s_ampa=0
Pe.s_gaba=0
Pe.s_nmda=0
Pe.x=0

Pi=NeuronGroup(inhib_n, eqs_i, threshold=Vth_i, reset=Vr_i, refractory=tr_i, order=2)
Pi.v=El_i
Pi.s_ext=0
Pi.s_ampa=0
Pi.s_gaba=0

#3 excitatory subgroups: 1 & 2 are selective to motion, ns not
Pe_1=Pe.subgroup(N1)
Pe_2=Pe.subgroup(N1)
Pe_ns=Pe.subgroup(excit_n-2*N1)

PG1=PoissonGroup(N1, lambda t:stimulus(t))
PG2=PoissonGroup(N1, lambda t:stimulus(t))
PGns=PoissonGroup(excit_n-2*N1, fext)
PGi=PoissonGroup(inhib_n, fext)

# Creating connections
print "Creating static conections ..."

selfnmda=IdentityConnection(Pe, Pe, 'x', weight=1.0, delay=0.5*ms)

Cp1=IdentityConnection(PG1, Pe_1, 's_ext', weight=1.0)
Cp2=IdentityConnection(PG2, Pe_2, 's_ext', weight=1.0)
Cpns=IdentityConnection(PGns, Pe_ns, 's_ext', weight=1.0)
Cpi=IdentityConnection(PGi, Pi, 's_ext', weight=1.0)

Cie=Connection(Pi, Pe, 's_gaba', weight=1.0, delay=0.5*ms)
Cii=Connection(Pi, Pi, 's_gaba', weight=1.0, delay=0.5*ms)
Cei=Connection(Pe, Pi, 's_ampa', weight=1.0, delay=0.5*ms)
Cee=Connection(Pe, Pe, 's_ampa', weight=structure, delay=0.5*ms)

for i in range(len(Pi)):
  Cii[i, i]=0

Wee=zeros((excit_n, excit_n))
for i in range(len(Pe)):
  for j in range(len(Pe)):
   Wee[i, j]=structure(i, j)

Wei=ones((excit_n, inhib_n))
Wei=transpose(Wei)

@network_operation(simulation_clock, when='start')
def f(simulation_clock):
    s_NMDA_full=transpose(array([Pe.s_nmda]))   # get the s_NMDA values of the excitatory neurons
    Pe.s_tot=transpose(dot(Wee, s_NMDA_full))  # Calculate the new s_tot values based on the weights for  the excitatory neurons
    Pi.s_tot=transpose(dot(Wei, s_NMDA_full))  # Calculate the new s_tot values based on the weights for the inhibitory neurons

# initiating monitors
M=SpikeMonitor(Pe)
#N = SpikeMonitor(Pi)

print "Running ..."
run(simtime, report='text')

print "Simulation finished."
raster_plot(M)
#raster_plot(N)
show()
