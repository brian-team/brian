from brian import *
from time import time

def f(N=100000, compile=False, freeze=False):
    clear(False)
    reinit_default_clock()
    eqs = '''
    dV/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
    dW/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
    dW2/dt = cos(2*pi*t/(100*ms))/(10*ms) : 1
    '''
    G = NeuronGroup(N, eqs, compile=compile, freeze=freeze)
    start = time()
    run(100*ms)
    return time()-start

print f(), '()'
print f(compile=True), '(compile)'
print f(freeze=True), '(freeze)'
print f(compile=True, freeze=True), '(compile, freeze)'