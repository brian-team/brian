'''
Simple pickling example with a small network.
This script looks for a previous data file
and if it finds one reads in saved network parameters
and connection weights.
Network parameters and connection weights are saved
at the end of a run.
'''
from brian import *
import pickle

# Look for an import file - if not there start
# a new empty one

try:
    imported_data = pickle.load(open('./saved.pickle', 'rb'))
    defaultclock.t = imported_data['t']
    print 'Starting from saved progress at time', defaultclock.t

except IOError:
    print 'No data to import, starting new run'
    imported_data = {}

# function which saves the network and connection
# data using pickle.

def save_progress():
    s = str(int((defaultclock.t+.5*ms)/second))
    imported_data['P._S'] = P._S
    imported_data['t'] = defaultclock.t
    exc_inds = [Ce.W[i, :].ind for i in range(Ne)]
    exc_weights = [asarray(Ce.W[i, :]) for i in range(Ne)]
    inh_inds = [Ci.W[i, :].ind for i in range(Ni)]
    inh_weights = [asarray(Ci.W[i, :]) for i in range(Ni)]
    imported_data['exc_inds'] = exc_inds
    imported_data['exc_weights'] = exc_weights
    imported_data['inh_inds'] = inh_inds
    imported_data['inh_weights'] = inh_weights
    pickle.dump(imported_data, open('./saved.pickle', 'wb'), -1)
    print 'Saved progress up to time', s, 'second'

# test network function using a different clock
# from default
@network_operation(clock=EventClock(t=defaultclock.t, dt=1000*msecond))
def test_func():
    print "Executing test_func at ", defaultclock.t


# Network parameters
Ne = 200
Ni = 50

# Model equations
eqs='''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''

# Create Neuron group
P=NeuronGroup(Ne+Ni,model=eqs,threshold=-50*mV,reset=-60*mV)
P.v=-60*mV
Pe=P.subgroup(Ne)
Pi=P.subgroup(Ni)

# If initialising from previous data then read in Neuron
# group data
if imported_data:
    P._S[:] = imported_data['P._S']

# Create connections

Ce=Connection(Pe,P,'ge')
Ci=Connection(Pi,P,'gi')

# If initialising from previous data then read in Connection
# data, otherwise connect randomly

if imported_data:
    exc_inds = imported_data['exc_inds']
    exc_weights = imported_data['exc_weights']
    inh_inds = imported_data['inh_inds']
    inh_weights = imported_data['inh_weights']
    for i in xrange(Ne):
        Ce[i, exc_inds[i]] = exc_weights[i]
    for i in xrange(Ni):
        Ci[i, inh_inds[i]] = inh_weights[i]
else:
    Ce.connect_random(Pe, P, 0.02,weight=1.62*mV)
    Ci.connect_random(Pi, P, 0.02,weight=-9*mV)


# Do the run
M=SpikeMonitor(P)

run(4000*ms)

# Save progress after run
save_progress()

# show what we've computed so far...
raster_plot(M)
show()
