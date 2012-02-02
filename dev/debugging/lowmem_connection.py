from brian import *
import gc
set_global_preferences(useweave=True, usecodegen=True)

N = 400
p = 0.7

G = PoissonGroup(N, rates=10*Hz)
H = NeuronGroup(N, 'V:1')
C = Connection(G, H, sparseness=p,
               #use_minimal_indices=False,
               delay=(0*ms, 1*ms),
               )

nsynapses = C.W.nnz

print 'Expected runtime memory size (orig):', (nsynapses*20)/1024**2, 'MB'
print 'Expected runtime memory size (new):', (nsynapses*16)/1024**2, 'MB'

run(1*ms)
gc.collect()
run(10*second, report='stderr')
