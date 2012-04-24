from brian import *
from brian.experimental.codegen2 import *
import numpy
import time
import numpy.random as nrandom
import random as prandom
try:
    import pycuda
except ImportError:
    pycuda = None    
log_level_info()

##### TESTING PARAMETERS
#from vectorise_over_postsynaptic_offset import *
#from vectorise_over_spiking_synapses import *
#from double_vectorise_over_postsynoff_targetidx_blocked import *
from double_vectorise_over_spsyn_targetidx_blocked import *
use_gpu = True
parameters = dict(use_atomic=True)
do_plot = False

##### PROFILING CODE

if use_gpu:
    Conn = GPUConnection
    language = GPULanguage(force_sync=do_plot)
else:
    Conn = Connection
    language = CLanguage()
    parameters = {}

eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
'''
threshold = 'v > -50*mV'
reset = '''
v = -60*mV
'''

nrandom.seed(213213)
prandom.seed(343831)

P = NeuronGroup(4000, eqs, threshold=threshold, reset=reset)
P.v = -60 * mV + 10 * mV * rand(len(P))
Pe = P.subgroup(3200)
Pi = P.subgroup(800)

P._state_updater = CodeGenStateUpdater(P, euler, language, clock=P.clock)
P._threshold = CodeGenThreshold(P, threshold, language)
P._resetfun = CodeGenReset(P, reset, language)

Ce = Conn(Pe, P, 'ge', weight=1.62*mV, sparseness=0.02, **parameters)
Ci = Conn(Pi, P, 'gi', weight=-9*mV, sparseness=0.02, **parameters)

if do_plot:
    M = SpikeMonitor(P)
    Mv = StateMonitor(P, 'ge', record=arange(100))
else:
    M = SpikeCounter(P)

if use_gpu:
    language.gpu_man.copy_to_device(True)

run(10*ms)
#exit()

start = time.time()
run(0.1*second-10*ms, report='stderr')
end = time.time()

if use_gpu:
    print 'Using GPU'
    for k, v in parameters.items():
        print '    '+k+':', v
else:
    print 'Using CPU'
correct_nspikes = 2443 # copied from CPU run
print 'Time:', end-start
print 'Num spikes:', M.nspikes, 'should be', correct_nspikes
if M.nspikes==correct_nspikes:
    print 'Success!'
else:
    print 'FAILED!'  

if do_plot:
    subplot(211)
    raster_plot(M)
    subplot(212)
    Mv.plot()
    show()
