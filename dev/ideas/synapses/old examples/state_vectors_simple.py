import dev.ideas.synapses.statevectors as sv
import numpy as np

reload(sv)

v = sv.ConstructionSparseStateVector(4, (np.int8, np.int8, np.float64, np.float64))

nvalues = 52

values = []
values.append(np.arange(nvalues, dtype = np.int8))
values.append(np.arange(nvalues, dtype = np.int8)[::-1])
values.append(np.array(np.random.rand(nvalues), dtype = np.float64))
values.append(np.array(np.random.rand(nvalues), dtype = np.float64))

v.append(values)

v.append(values)

v.append(values)

vcomp = v.compress()

