# Reward - based STDP based on code from nemo136
# posted on Brian Support Google Group

from brian import *

#---------------------------------------------------
# Variable declarations
#---------------------------------------------------


# Model variables and equations
dt=1*ms
tau=10*ms
Vr=-70*mV
Vl=-55*mV
lif_eqs = '''
dVm/dt=(Vl-Vm)/tau : volt
K : volt
'''

# Basic STDP variables
tau_pre=10*ms
tau_post=10*ms
gmax=1
dA_pre=.001
dA_post=-dA_pre*tau_pre/tau_post*2.5


#-------------------------------------------------------
# Set up neuron groups
#-------------------------------------------------------

# Input group of Poisson neurons
groupi=PoissonGroup(100,rates=linspace(0.1*Hz,10*Hz,100))

# Main neuron population, split into two subgroups

groupa = NeuronGroup(100, model=lif_eqs, threshold=-55*mV,
                     reset=Vr,max_delay=6*ms,)
suba=groupa[0:50]
subb=groupa[50:100]

#-------------------------------------------------------
# Create connections
#-------------------------------------------------------

# Connect input to group a - this inputs to Voltage in main eqs

input_signal = Connection(groupi,suba, 'Vm', weight=0.2*mV,
                          sparseness=0.6,delay=4*ms)

# Connect group a to b - this inputs to Voltage in main eqs

synapses = Connection(suba,subb, 'Vm', weight=0.1*mV,
                      sparseness=0.7,delay=3*ms,structure='sparse', column_access=True)

# Make trace connection which is a copy of synapses connection

trace = Connection(suba,subb, 'K', weight = synapses.W)

trace.compress()

T_matrix = trace.W

#-------------------------------------------------------
# Set up STDP on trace connection to collect the weight
# changes rather than apply them immediately
#-------------------------------------------------------

eqs_stdp='''
dA_pre/dt=-A_pre/tau_pre : 1
dA_post/dt=-A_post/tau_post : 1
'''
stdp=STDP(trace,eqs=eqs_stdp,pre='A_pre+=dA_pre;w+=A_post',
                  post='A_post+=dA_post;w+=A_pre',wmax=gmax)


def update_trace(spikes):
        if len(spikes)>0:
            # Apply reward to STDP weights which have been
            # collected in trace
            for j in spikes:
                T_matrix[:,j] *= 0.05
#            for i in xrange(len(trace.W.get_col(0))):
#                for j in spikes:
#                    if not isinstance(T_matrix[i,j],int):
#                        T_matrix[i,j] *= 0.05


#---------------------------------------------------------
# Create monitors
#---------------------------------------------------------


# Custom SpikeMonitor for applying reward

Sa1 = SpikeMonitor(suba,function=update_trace)


# Set clock
defaultclock.dt = 1.0*ms

# Initialise the Voltage to resting voltage, Vr

groupa.Vm=Vr

# Run

run(500*msecond)
