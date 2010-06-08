'''
Ideas for automating the work of parallelising a network

* Given a network object net, work through each of the objects
  adapting them all for the expansion of some parameter. This
  is nice because it doesn't require the user to write any new
  code just add in an extra auto-parallelisation line. The
  difficult thing is that it would require adapting the objects
  so that as well as creating them they also store a copy of
  their parameters in such a way that they can be recreated
  (e.g. extending a neurongroup from N=1 to N=1000). This is
  tricky at the moment.
* Given a function that defines a simulation to be run, rewrite
  the code automatically. One nice way to do this is to replace
  each NeuronGroup object in the global namespace of the function
  with a new type of NeuronGroup object that automatically adapts
  it to the parameter set given (multiplies N by number of
  different parameters, etc.). This is also nice since it means
  the user just has to wrap their code in a function. The
  difficult thing here is that referring to the objects
  defined by the function is tricky - how should it be done?
'''

from brian import *
import inspect
from copy import copy

def printargs(*args, **kwds):
    print args
    print kwds


class AutoParallelNetwork(Network):
    def __init__(self, makenet, params):
        self.apn_params = params
        self.apn_makenet = makenet
        Network.__init__(self)

        makenet_globals = copy(makenet.func_globals)
        makenet_globals['NeuronGroup'] = printargs
        newfuncspace = {}
        exec inspect.getsource(makenet) in makenet_globals, newfuncspace
        exec makenet.func_name + '()' in makenet_globals, newfuncspace

def makenet():
    global M
    Vr = -70 * mV
    Vt = -55 * mV
    El = -54 * mV
    eqs = '''
        dV/dt = -(V-El)/tau : volt
        tau : second
        '''
    model = Model(equations=eqs, threshold=Vt, reset=Vr)
    G = NeuronGroup(1, model)
#    spikes = linspace(10*ms,100*ms,25)
#    input = MultipleSpikeGeneratorGroup([spikes])
#    C = Connection(input, G)
#    C[0,0] = 5*mV
    M = StateMonitor(G, 'V', record=True)
#    G.V = Vr
#    G.tau = 10*ms
#    net = MagicNetwork(verbose=False)
#    return net

makenet()

print 'made net'

anet = AutoParallelNetwork(makenet,
        [
         ('G', 'tau', [5 * ms, 10 * ms, 15 * ms])
        ]
        )

#net = makenet()
#net.run(100*ms)
#
#plot(M.times/ms,M[0]/mV)
#show()
