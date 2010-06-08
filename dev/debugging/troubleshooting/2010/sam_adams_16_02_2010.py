from brian import *
from brian.library.random_processes import *

# General variables
N = 100
RN_idx = 10 # the reinforced neuron
connP = 0.02 # Connection probability for reduced noise connections
connP_EE_scale = 1.0 # Scaling factor for EE connection
connP_IE_scale = 1.2 # Scaling factor for IE connection

# Parameters for conductance based Leaky Integrate and Fire Neuron

Cm = 0.3 * nfarad # Membrane capacitance
Rm = 100 * Mohm # Membrane resistance
# Time constants
taus = 5 * msecond # Excitatory/Inhibitory synapse time constant
# Potential settings
Ee = 0 * mV # Excitatory synapse reversal potential 
Ei = -75 * mV # Inhibitory synapse reversal potential
Vthresh = -59 * mV # Threshold Potential
Vresting = -70 * mV # Resting Potential (V(0))
Vreset = -70 * mvolt # Reset potential
# Delays
Trefrac = 5 * msecond # Refractory period
Tdelay = 1 * msecond # Synaptic time delay

# STDP variables
tau_pre = 30 * msecond
tau_post = 30 * msecond
Ap = 0.01
Am = -Ap * tau_pre / tau_post * 1.05 # approx -0.01

# RM-STDP specific variables

TauElig = 0.4 * second # Tau for eligibility trace
ArPos = 1.379 # Alpha plus for reward kernel
ArNeg = 0.27 # Alpha neg for reward kernel
TauRPos = 0.2 * second # Tau plus for reward kernel
TauRNeg = 1.0 * second # Tau neg for reward kernel
TDelay = 0.2 * second # Reward delay
last_rn_spike = 0.0 * second # Time of previous reinforced neuron spike
current_rn_spike = 0.0 * second # Time of most recent reinforced neuron spike
current_reward = 0.0 # Global variable for reward signal

# some arrays for plotting an eligibility trace
f_times = []
f_values = []


# O-U Noise equations

eqs_Enoise = OrnsteinUhlenbeck('Enoise', mu=0.012 * usiemens, sigma=0.003 * usiemens, tau=2.7 * ms)
eqs_Inoise = OrnsteinUhlenbeck('Inoise', mu=0.057 * usiemens, sigma=0.0066 * usiemens, tau=10.5 * ms)

# Equations for eligibility trace
# f is the eligibility kernel function
# c is the eligibility trace which is
# (STDP wt change * f)

elig_eqs = Equations('''
last_time:second
tc_elig = (t - last_time) *(1.0/(TauElig + TDelay)):1
f = 0.5 * tc_elig * exp(1.0 - tc_elig):1
c:1
''')

# Equations

eqs_full = Equations('''
dV/dt = (V - Vresting)*(1./(Cm*Rm)) + (ge*(V - Ee))*(1./Cm) + (gi*(V - Ei))*(1./Cm) - (0.2*Enoise*(V - Ee))*(1./Cm) - (0.2*Inoise*(V - Ei))*(1./Cm): volt
dge/dt = -ge*(1./taus) : siemens
dgi/dt = -gi*(1./taus) : siemens
''')


eqs_main = eqs_full + eqs_Enoise + eqs_Inoise + elig_eqs

# Main neuron groups

Pe = NeuronGroup(N / 2, model=eqs_main, threshold=Vthresh, reset=Vreset, refractory=Trefrac, implicit=True)
Pe.V = Vresting
Pe.ge = 0 * mV
Pe.gi = 0 * mV
Pe.last_time = 0.0 * second

Pi = NeuronGroup(N / 2, model=eqs_main, threshold=Vthresh, reset=Vreset, refractory=Trefrac, implicit=True)
Pi.V = Vresting
Pi.ge = 0 * mV
Pi.gi = 0 * mV
Pi.last_time = 0.0 * second

# Synapse weight equation parameters

Wscale = 0.8
WExcScale = 1.0
WInhScale = 1.4
taum = Rm * Cm

we = ((Vthresh - Vresting) * WExcScale * Wscale) / ((Ee - Vresting) * Rm * taus / (taum - taus) * (((taus / taum) ** (taus / (taum - taus))) - ((taus / taum) ** (taum / (taum - taus)))))
wi = ((Vthresh - Vresting) * WInhScale * Wscale) / ((Vresting - Ei) * Rm * taus / (taum - taus) * (((taus / taum) ** (taus / (taum - taus))) - ((taus / taum) ** (taum / (taum - taus)))))

# Maximum weight allowable during STDP is set to 2 * initial excitatory weight

wMax = we * 2.0

# Make a neuron subgroup which is the reinforced neuron

RN = Pe[RN_idx]

Cee = Connection(Pe, Pe, 'ge', weight=we, sparseness=connP * connP_EE_scale, delay=Tdelay)
Cie = Connection(Pe, Pe, 'gi', weight=wi, sparseness=connP * connP_IE_scale, delay=Tdelay)


# Custom network operations to collect the eligibility
# trace for reinforced neuron for plottingNameError: global name 'TauElig' is not defined

@network_operation
def save_elig():
    f_times.append(defaultclock.t)
    f_values.append(Pe.f[RN_idx])


M = SpikeMonitor(Pe)
run(200 * msecond)
raster_plot(M)

# Create plot of elig values

figure()
plot(elig_times, elig, 'k')
xlabel("Time/ms")
ylabel("Elig")
suptitle('Eligibility Trace', fontsize=12)


