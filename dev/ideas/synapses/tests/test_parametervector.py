from brian import *
import dev.ideas.synapses.synapses as syn
reload(syn)

existing_synapses = np.array([[1,2,3], [4,5,6]])
groups_shape = (4, 7)
data = np.zeros(existing_synapses.shape[1])

pv = syn.ParameterVector(data, groups_shape, existing_synapses)


print 'initial', pv.data
pv[1,:] = 3


print 'modified', pv.data
print data

