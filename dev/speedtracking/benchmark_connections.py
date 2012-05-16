from vbench.benchmark import Benchmark
from datetime import datetime

common_setup = """
from brian import *
"""

python_only_setup = """
set_global_preferences(useweave=False, usecodegen=False, usecodegenweave=False)
"""

weave_setup = """
set_global_preferences(useweave=True,
                       gcc_options=['-march=native', '-ffast-math', '-O3'])
"""

codegen_setup = """
set_global_preferences(usecodegenweave=True, usecodegen=True,
                       gcc_options=['-march=native', '-ffast-math', '-O3'])
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
net.run(defaultclock.dt)
"""

statement = "net.run(1 * second)"

# Code generation was introduced here
start_codegen = datetime(2010, 2, 6)

# Sparse matrices
bench_sparse = Benchmark(statement,
                         common_setup + python_only_setup + \
                         setup_template % {'neurons' : 10,
                                           'structure' : 'sparse'},
                         name='sparse connection matrix (10x10)')

bench_sparse100 = Benchmark(statement,
                            common_setup + python_only_setup + \
                            setup_template % {'neurons' : 100,
                                              'structure' : 'sparse'},
                            name='sparse connection matrix (100x100)')
bench_sparse100w = Benchmark(statement,
                             common_setup + weave_setup + \
                             setup_template % {'neurons' : 100,
                                               'structure' : 'sparse'},
                             name='sparse connection matrix (100x100) with weave')
bench_sparse100wc = Benchmark(statement,
                              common_setup + weave_setup + codegen_setup +\
                              setup_template % {'neurons' : 100,
                                                'structure' : 'sparse'},
                              name='sparse connection matrix (100x100) with weave + codegen',
                              start_date=start_codegen)
# Dynamic matrices
# Set a start date here because the benchmark fails for earlier revisions
start_dynamic = datetime(2010, 2, 4)
bench_dynamic = Benchmark(statement,
                          common_setup + python_only_setup + \
                          setup_template % {'neurons' : 5,
                                            'structure' : 'dynamic'},
                          name='dynamic connection matrix (5x5)',
                          start_date=start_dynamic)
bench_dynamic50 = Benchmark(statement,
                            common_setup + python_only_setup + \
                            setup_template % {'neurons' : 50,
                                              'structure' : 'dynamic'},
                            name='dynamic connection matrix (50x50)',
                            start_date=start_dynamic)
bench_dynamic50w = Benchmark(statement,
                             common_setup + weave_setup + \
                             setup_template % {'neurons' : 50,
                                               'structure' : 'dynamic'},
                             name='dynamic connection matrix (50x50) with weave',
                             start_date=start_dynamic)

# Dense matrices
bench_dense = Benchmark(statement,
                        common_setup + python_only_setup + \
                        setup_template % {'neurons' : 10,
                                          'structure' : 'dense'},
                        name='dense connection matrix (10x10)')
bench_dense100 = Benchmark(statement,
                           common_setup + python_only_setup + \
                           setup_template % {'neurons' : 100,
                                             'structure' : 'dense'},
                           name='dense connection matrix (100x100)')
bench_dense100w = Benchmark(statement,
                            common_setup + weave_setup + \
                            setup_template % {'neurons' : 100,
                                              'structure' : 'dense'},
                            name='dense connection matrix (100x100) with weave')
bench_dense100wc = Benchmark(statement,
                             common_setup + weave_setup + codegen_setup +\
                             setup_template % {'neurons' : 100,
                                              'structure' : 'dense'},
                             name='dense connection matrix (100x100) with weave + codegen',
                             start_date=start_codegen)