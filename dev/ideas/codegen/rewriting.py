from brian import *
import sympy
from brian.optimiser import *
from brian.inspection import *

__all__ = ['rewrite_to_c_expression', 'sympy_rewrite']

pow = sympy.Function('pow')

def rewrite_pow(e):
    if not len(e.args):
        return e
    newargs = tuple(rewrite_pow(a) for a in e.args)
    if isinstance(e, sympy.Pow) and e.args[1]!=-1:
        return pow(*newargs)
    else:
        return e.new(*newargs)

def make_sympy_expressions(eqs):
    exprs = {}
    for name in eqs._diffeq_names+eqs._eq_names:
        exprs[name] = symbolic_eval(eqs._string[name])
    return exprs

def sympy_rewrite(s):
    return str(symbolic_eval(s))

def rewrite_to_c_expression(s):
    e = symbolic_eval(s)
    return str(rewrite_pow(e))

def generate_c_expressions(eqs):
    exprs = make_sympy_expressions(eqs)
    cexprs = {}
    for name, expr in exprs.iteritems():
        cexprs[name] = str(rewrite_pow(expr))
    return cexprs

if __name__=='__main__':
    if True:
        area=20000*umetre**2
        Cm=(1*ufarad*cm**-2)*area
        gl=(5e-5*siemens*cm**-2)*area
        El=-60*mV
        EK=-90*mV
        ENa=50*mV
        g_na=(100*msiemens*cm**-2)*area
        g_kd=(30*msiemens*cm**-2)*area
        VT=-63*mV
        # Time constants
        taue=5*ms
        taui=10*ms
        # Reversal potentials
        Ee=0*mV
        Ei=-80*mV
        we=6*nS # excitatory synaptic weight (voltage)
        wi=67*nS # inhibitory synaptic weight
        # The model
        eqs=Equations('''
        dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-g_na*(m*m*m)*h*(v-ENa)-g_kd*(n*n*n*n)*(v-EK))/Cm : volt 
        dm/dt = alpham*(1-m)-betam*m : 1
        dn/dt = alphan*(1-n)-betan*n : 1
        dh/dt = alphah*(1-h)-betah*h : 1
        dge/dt = -ge*(1./taue) : siemens
        dgi/dt = -gi*(1./taui) : siemens
        alpham = 0.32*(mV**-1)*(13*mV-v+VT)/(exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
        betam = 0.28*(mV**-1)*(v-VT-40*mV)/(exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
        alphah = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
        betah = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
        alphan = 0.032*(mV**-1)*(15*mV-v+VT)/(exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
        betan = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
        ''')
        cexprs = generate_c_expressions(eqs)
        for k, v in cexprs.iteritems():
            print k, ':', v
    if False:
        taum=20*msecond
        taue=5*msecond
        taui=10*msecond
        Ee=(0.+60.)*mvolt
        Ei=(-80.+60.)*mvolt
        eqs = Equations('''
        dv/dt = (-v+ge*(Ee-v)+gi*(Ei-v))*(1./taum) : volt
        dge/dt = -ge*(1./taue) : 1
        dgi/dt = -gi*(1./taui) : 1 
        ''')
        cexprs = generate_c_expressions(eqs)
        for k, v in cexprs.iteritems():
            print k, ':', v
    if False:
        x = sympy.Symbol('x')
        y = x**2 + 3*x    
        print y
        print rewrite_pow(y)