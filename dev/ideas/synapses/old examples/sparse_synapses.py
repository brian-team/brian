'''
Testing the ability to construct synapses with strings!
'''

from brian import *
import dev.ideas.synapses.synapses as syn
reload(syn)
log_level_debug()

gin = NeuronGroup(1000, model = 'dv/dt = -v/(5*ms) : 1')
gout = NeuronGroup(1000, model = 'dv/dt = -v/(5*ms) : 1')

# init synapses
print 'SPARSE SYNAPSES'
synapses = syn.Synapses(gin, gout, model = '''w : 1; z : 1''', pre = 'v += w')

synapses[:,:] = '''rand() < .03'''

print 'proportion of created synapses: ', float(len(synapses)) / (len(gin) * len(gout))


del synapses
synapses = syn.Synapses(gin, gout, model = '''w : 1; z : 1''', pre = 'v += w')

print 'MULTIPLE SYNAPSES'
synapses[:,:] = ''' (i == j)'''
# this is slow
