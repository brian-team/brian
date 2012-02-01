from vbench.benchmark import Benchmark
from datetime import datetime

common_setup = """
from brian import *
"""

setup_template = """
s = arange(%(neurons)d)
class thr(Threshold):
    def __call__(self, P):
        return s
G = NeuronGroup(%(neurons)d, 'V:1', threshold=thr())
H = NeuronGroup(%(neurons)d, 'V:1')
C = Connection(G, H, structure='%(structure)s' )
C.connect_full(G, H, weight=1)
net = Network(G, H, C)
net.prepare()
"""

statement = "run(1 * second)"

bench_sparse = Benchmark(statement,
                         common_setup + (setup_template % {'neurons' : 10, 'structure' : 'sparse'}),
                         name='sparse')
bench_dynamic = Benchmark(statement,
                          common_setup + (setup_template % {'neurons' : 5, 'structure' : 'dynamic'}),
                          name='dynamic')
bench_dense = Benchmark(statement,
                        common_setup + (setup_template % {'neurons' : 10, 'structure' : 'dense'}),
                        name='dense')

