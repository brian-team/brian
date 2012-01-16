'''
In this script I test the synapses initialization/construction of Synapses
'''
from brian import *
import dev.ideas.synapses.synapses as syn
reload(syn)
log_level_debug()

gin = NeuronGroup(10, model = 'dv/dt = -v/(5*ms) : 1')
gout = NeuronGroup(20, model = 'dv/dt = -v/(5*ms) : 1')

synapses = syn.Synapses(gin, gout, model = '''w : 1; z : 1''', pre = 'v += w + rand()')

synapses[0,:] = 2
print synapses
synapses.w[:,:] = 2.1
synapses.z[1,:] = 4
print synapses.w
print synapses.z

synapses.z[0, :, :] = arange(len(synapses)*2, dtype = np.float32)
print synapses.z

