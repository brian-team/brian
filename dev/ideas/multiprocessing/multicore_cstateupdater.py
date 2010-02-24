from brian import *
from brian.experimental.codegen.codegen_c import *
from brian.experimental.codegen.integration_schemes import *
import time
from scipy import weave
import numpy, scipy
from brian.experimental.codegen.c_support_code import *
from handythread import *
import re

class MulticoreCStateUpdater(StateUpdater):
    def __init__(self, eqs, scheme, clock=None, freeze=False, threads=1):
        self.threads = threads
        self.clock = guess_clock(clock)
        self.code_c = CCodeGenerator().generate(eqs, scheme)
        self.code_c = "Py_BEGIN_ALLOW_THREADS\n" + self.code_c + "Py_END_ALLOW_THREADS\n"
        code = ''
        for line in self.code_c.split('\n'):
            if line.startswith('double *') and line.endswith('num_neurons;'):
                line = line[:-1]+'+offset;'
            if line=='for(int _i=0;_i<num_neurons;_i++){':
                line = 'for(int _i=0; _i<num_neurons_segment; _i++){'
            code += line+'\n'
        self.code_c = code
        print self.code_c
        log_debug('brian.experimental.codegen.stateupdaters', 'C state updater code:\n'+self.code_c)
        self._weave_compiler = get_global_preference('weavecompiler')
        self._extra_compile_args = ['-O3']
        if self._weave_compiler=='gcc':
            self._extra_compile_args += ['-march=native']
        self._prepared = False
    def __call__(self, P):
        if not self._prepared:
            dt = P.clock._dt
            num_neurons = len(P)
            num_neurons_segment = num_neurons
            offset = 0
            _S = P._S
            t = P.clock._t
            weave.inline(self.code_c, ['_S', 'num_neurons', 'num_neurons_segment',
                                       'dt', 't', 'offset'],
                         support_code=c_support_code,
                         compiler=self._weave_compiler,
                         extra_compile_args=self._extra_compile_args,
                         #force=True,
                         )
            def makefunc(_S, offset, num_neurons, num_neurons_segment, dt):
                d = {'_S':_S, 'offset':int(offset), 'num_neurons':int(num_neurons),
                     'num_neurons_segment':int(num_neurons_segment), 'dt':dt}
                def f(t):
                    d['t'] = t
                    weave.inline(self.code_c, ['_S', 'num_neurons', 'num_neurons_segment',
                                               'dt', 't', 'offset'],
                                 local_dict=d,
                                 support_code=c_support_code,
                                 compiler=self._weave_compiler,
                                 extra_compile_args=self._extra_compile_args)
                return f
            pieces = array([len(P)/self.threads for _ in range(self.threads)], dtype=int)
            pieces[-1] += len(P)-sum(pieces)
            offsets = hstack((0, cumsum(pieces)))[:-1]
            self.funcs = [makefunc(_S, offset, num_neurons, piece, dt) for offset, piece in zip(offsets, pieces)]
            self._prepared = True
        else:
            t = P.clock._t
            parallel_map(lambda f:f(t), self.funcs, threads=self.threads)

if __name__=='__main__':
    from brian.library.ionic_currents import *
    
    N = 10000
    duration = 100*ms
    use_multicore = True
    numthreads = 2
    record_and_plot = False
    
    El=10.6*mV
    EK=-12*mV
    ENa=120*mV
    eqs=MembraneEquation(1*uF)+leak_current(.3*msiemens,El)
    eqs+=K_current_HH(36*msiemens,EK)+Na_current_HH(120*msiemens,ENa)
    eqs+=Current('I:amp')
    eqs.prepare()

    neuron=NeuronGroup(N,eqs,implicit=True,freeze=True)
    
    if use_multicore:
        neuron._state_updater = MulticoreCStateUpdater(eqs, exp_euler_scheme,
                                                       threads=numthreads)
    
    if record_and_plot:
        trace=StateMonitor(neuron,'vm',record=True)
    
    neuron.I=10*uA

    net = MagicNetwork()
    net.run(1*ms)

    start = time.time()
    net.run(duration)
    end = time.time()
    
    print end-start

    if record_and_plot:
        trace.plot()
        show()
