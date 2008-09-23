from brian import *

def custom_reset(P, spikes):
    P.V_[spikes]=0
    # plus do other stuff here...

gen = MultipleSpikeGeneratorGroup([[10*ms, 20*ms, 30*ms]])
inp = NeuronGroup(1, model='V:1', threshold=0.5, reset=custom_reset)
outp = NeuronGroup(1, model='V:1', threshold=0.5, reset=0.0)
C_gen_inp = IdentityConnection(gen, inp)
C = IdentityConnection(inp, outp)
Mgen = SpikeMonitor(gen)
Minp = SpikeMonitor(inp)
Moutp = SpikeMonitor(outp)
run(100*ms)
for M in [Mgen, Minp, Moutp]:
    print M.spikes