from brian import *
from languages import *
from integration import *
from symbols import *

class ValueSymbol(Symbol):
    '''
    This class is simply used to replace some known symbols with constant
    values. In :func:`make_c_integrator` it is just used for setting the value
    of dt. We just change :meth:`Symbol.read` to return ``'0.001'`` or whatever.
    '''
    supported_languages = ['c']
    def __init__(self, name, language, value):
        Symbol.__init__(self, name, language)
        if isinstance(value, float):
            value = float(value)
        if isinstance(value, int):
            value = int(value)
        self.value = value
    def read(self):
        return str(self.value)
    
class TimeSymbol(Symbol):
    '''
    This symbol is used to redefine the Brian symbol 't' to use a different
    name (like T or time) and to potentially give it a different scale (e.g.
    ms instead of second). We simply replace :meth:`Symbol.read` with
    ``'(scale*name)'``, the parentheses so that it works as part of a larger
    expression.
    '''
    supported_languages = ['c']
    def __init__(self, name, language, timeunit):
        Symbol.__init__(self, name, language)
        self.timeunit = timeunit
    def read(self):
        if self.timeunit is second:
            return self.name
        else:
            return '('+str(float(self.timeunit))+'*'+self.name+')'

@check_units(dt=second)
def make_c_integrator(eqs, method, dt, values=None, scalar='double',
                      timeunit=second, timename='t'):
    '''
    Gives C/C++ format code for the integration step of a differential equation.
    
    ``eqs``
        The equations, can be an :class:`brian.Equations` object or a multiline
        string in Brian equations format.
    ``method``
        The integration method, typically :func:`euler`, :func:`rk2` or
        :func:`exp_euler`, although you can pass your own integration method,
        see :func:`make_integration_step` for details.
    ``dt``
        The value of the timestep dt (in Brian units, e.g. ``0.1*ms``)
    ``values``
        Optional, dictionary of mappings variable->value, these values will
        be inserted into the generated code.
    ``scalar``
        By default it is ``'double'`` but if you want to use float as your
        scalar type, set this to ``'float'``.
    ``timename``
        The name of the time variable (if used). In Brian this is 't', but
        you can change it to 'T' or 'time' or whatever. This can be used if you
        want users to specify time in Brian form ('t') but the context in which
        this code will be used (e.g. another simulator) specifies time with a
        different variable name (e.g. 'T').
    ``timeunit``
        The unit of the time variable, scaled because Brian expects time to be
        in seconds.
        
    Returns a triple ``(code, vars, params)``:
    
    ``code``
        The C/C++ code to perform the update step (string).
    ``vars``
        A list of variable names.
    ``params``
        A list of per-neuron parameter names.
    '''
    if values is None:
        values = {}
    if not isinstance(eqs, Equations):
        # update the namespace so that names like 'sin', 'exp' are included
        ns = globals()
        ns.update(locals())
        ns.update(values)
        eqs = Equations(eqs, **ns)
    language = CLanguage(scalar=scalar)
    symbols = {
        'dt': ValueSymbol('dt', language, float(dt)),
        't': TimeSymbol(timename, language, timeunit),
        }
    steps = make_integration_step(method, eqs)
    code = '\n'.join([s.convert_to(language, symbols=symbols) for s in steps])
    vars = eqs._diffeq_names_nonzero
    allvars = eqs._diffeq_names
    params = [p for p in allvars if not p in vars]
    return code, vars, params

if __name__=='__main__':
    if 1:
        eqs = '''
        dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
        dge/dt = -ge/(5*ms) : volt
        dgi/dt = -gi/(10*ms) : volt
        '''
        code, vars, params = make_c_integrator(eqs, method=euler, dt=0.1*ms)
        print code
    if 0:
        eqs = '''
        dv/dt = (ge*sin(2*pi*500*Hz*t)+gi-(v+49*mV))/(20*ms) : volt
        dge/dt = -ge/tau : volt
        dgi/dt = -gi/tau_i : volt
        tau : second
        '''
        code, vars, params = make_c_integrator(eqs,
                method=euler,
                dt=0.1*ms,
                values={'tau_i':10*ms},
                scalar='float',
                timeunit=1*ms,
                timename='T')
        print 'vars =', vars
        print 'params =', params
        print
        print code
