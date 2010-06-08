# ----------------------------------------------------------------------------------
# Copyright ENS, INRIA, CNRS
# Contributors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
# 
# Brian is a computer program whose purpose is to simulate models
# of biological neural networks.
# 
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# ----------------------------------------------------------------------------------
# 
'''
Reset mechanisms
'''

__all__ = ['Reset', 'VariableReset', 'Refractoriness', 'NoReset', 'FunReset',
         'CustomRefractoriness', 'SimpleCustomRefractoriness', 'StringReset',
         'select_reset']

from numpy import where, zeros
from units import *
from clock import *
import inspect
import re
import numpy
from inspection import *
from utils.documentation import flattened_docstring
from globalprefs import *
from log import *
CReset = PythonReset = None

def select_reset(expr, eqs, level=0):
    '''
    Automatically selects the appropriate Reset object from a string.
    
    Matches the following patterns if expr is a one liner:
    
    var_name = const : Reset
    var_name = var_name : VariableReset
    others : StringReset
    '''
    # plan:
    # - strip it and see if it is one line, if not select StringReset
    # - see if it matches A = B, if not select StringReset
    # - check if A, B both match diffeq variable names, and if so
    #   select VariableReset
    # - check that A is a variable name, if not select StringReset
    # - extract all the identifiers from B, and if none of them are
    #   callable, assume it is a constant, try to eval it and then use
    #   Reset. If not, or if eval fails, use StringReset
    # This misses the case of e.g. V=10*mV*exp(1) because exp will be
    # callable, but in general a callable means that it could be
    # non-constant.
    global CReset, PythonReset
    use_codegen = get_global_preference('usecodegen') and get_global_preference('usecodegenreset')
    use_weave = get_global_preference('useweave') and get_global_preference('usecodegenweave')
    if use_codegen:
        if CReset is None:
            from experimental.codegen.reset import CReset, PythonReset
        if use_weave:
            log_warn('brian.reset', 'Using codegen CReset')
            return CReset(expr, level=level + 1)
        else:
            log_warn('brian.reset', 'Using codegen PythonReset')
            return PythonReset(expr, level=level + 1)
    expr = expr.strip()
    if '\n' in expr:
        return StringReset(expr, level=level + 1)
    eqs.prepare()
    ns = namespace(expr, level=level + 1)
    s = re.search(r'\s*(\w+)\s*=(.+)', expr)
    if not s:
        return StringReset(expr, level=level + 1)
    A = s.group(1)
    B = s.group(2).strip()
    if A not in eqs._diffeq_names:
        return StringReset(expr, level=level + 1)
    if B in eqs._diffeq_names:
        return VariableReset(B, A)
    vars = get_identifiers(B)
    all_vars = eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t']
    for v in vars:
        if v not in ns or v in all_vars or callable(ns[v]):
            return StringReset(expr, level=level + 1)
    try:
        val = eval(B, ns)
    except:
        return StringReset(expr, level=level + 1)
    return Reset(val, A)


class Reset(object):
    '''
    Resets specified state variable to a fixed value
    
    **Initialise as:** ::
    
        R = Reset([resetvalue=0*mvolt[, state=0]])
        
    with arguments:
    
    ``resetvalue``
        The value to reset to.
    ``state``
        The name or number of the state variable to reset.

    This will reset all of the neurons that have just spiked. The
    given state variable of the neuron group will be set to value
    ``resetvalue``.
    '''
    def __init__(self, resetvalue=0 * mvolt, state=0):
        self.resetvalue = resetvalue
        self.state = state
        self.statevectors = {}

    def __call__(self, P):
        '''
        Clamps membrane potential at reset value.
        '''
        V = self.statevectors.get(id(P), None)
        if V is None:
            V = P.state_(self.state)
            self.statevectors[id(P)] = V
        V[P.LS.lastspikes()] = self.resetvalue

    def __repr__(self):
        return 'Reset ' + str(self.resetvalue)


class StringReset(Reset):
    '''
    Reset defined by a string
    
    Initialised with arguments:
    
    ``expr``
        The string expression used to reset. This can include 
        multiple lines or statements separated by a semicolon.
        For example, ``'V=-70*mV'`` or ``'V=-70*mV; Vt+=10*mV'``.
        Some standard functions are provided, see below.
    ``level``
        How many levels up in the calling sequence to look for
        names in the namespace. Usually 0 for user code.
    
    Standard functions for expressions:
    
    ``rand()``
        A uniform random number between 0 and 1.
    ``randn()``
        A Gaussian random number with mean 0 and standard deviation 1.
    
    For example, these could be used to implement an adaptive
    model with random reset noise with the following string::
    
        E -= 1*mV
        V = Vr+rand()*5*mV
    '''
    def __init__(self, expr, level=0):
        expr = flattened_docstring(expr)
        self._namespace, unknowns = namespace(expr, level=level + 1, return_unknowns=True)
        self._prepared = False
        self._expr = expr
        class Replacer(object):
            def __init__(self, func, n):
                self.n = n
                self.func = func
            def __call__(self):
                return self.func(self.n)
        self._Replacer = Replacer

    def __call__(self, P):
        if not self._prepared:
            unknowns = [var for var in P.var_index if isinstance(var, str)]
            expr = self._expr
            for var in unknowns:
                expr = re.sub("\\b" + var + "\\b", var + '[_spikes_]', expr)
            self._code = compile(expr, "StringReset", "exec")
            self._vars = unknowns
        spikes = P.LS.lastspikes()
        self._namespace['_spikes_'] = spikes
        self._namespace['rand'] = self._Replacer(numpy.random.rand, len(spikes))
        self._namespace['randn'] = self._Replacer(numpy.random.randn, len(spikes))
        for var in self._vars:
            self._namespace[var] = P.state(var)
        exec self._code in self._namespace

    def __repr__(self):
        return "String reset"


class VariableReset(Reset):
    '''
    Resets specified state variable to the value of another state variable
    
    Initialised with arguments:
    
    ``resetvaluestate``
        The state variable which contains the value to reset to.
    ``state``
        The name or number of the state variable to reset.

    This will reset all of the neurons that have just spiked. The
    given state variable of the neuron group will be set to
    the value of the state variable ``resetvaluestate``.
    '''
    def __init__(self, resetvaluestate=1, state=0):
        self.resetvaluestate = resetvaluestate
        self.state = state
        self.resetstatevectors = {}
        self.statevectors = {}

    def __call__(self, P):
        '''
        Clamps membrane potential at reset value.
        '''
        V = self.statevectors.get(id(P), None)
        if V is None:
            V = P.state_(self.state)
            self.statevectors[id(P)] = V
        Vr = self.resetstatevectors.get(id(P), None)
        if Vr is None:
            Vr = P.state_(self.resetvaluestate)
            self.resetstatevectors[id(P)] = Vr
        lastspikes = P.LS.lastspikes()
        V[lastspikes] = Vr[lastspikes]

    def __repr__(self):
        return 'VariableReset(' + str(self.resetvaluestate) + ', ' + str(self.state) + ')'


class FunReset(Reset):
    '''
    A reset with a user-defined function.
    
    **Initialised as:** ::
    
        FunReset(resetfun)
    
    with argument:
    
    ``resetfun``
        A function ``f(G,spikes)`` where ``G`` is the
        :class:`NeuronGroup` and ``spikes`` is an array of
        the indexes of the neurons to be reset.
    '''
    def __init__(self, resetfun):
        self.resetfun = resetfun

    def __call__(self, P):
        self.resetfun(P, P.LS.lastspikes())


class Refractoriness(Reset):
    '''
    Holds the state variable at the reset value for a fixed time after a spike.

    Initialised with arguments:
    
    ``resetvalue``
        The value to reset and hold to.
    ``period``
        The length of time to hold at the reset value. If using variable
        refractoriness, this is the maximum period.
    ``state``
        The name or number of the state variable to reset and hold.
    '''
    @check_units(period=second)
    def __init__(self, resetvalue=0 * mvolt, period=5 * msecond, state=0):
        self.period = period
        self.resetvalue = resetvalue
        self.state = state
        self._periods = {} # a dictionary mapping group IDs to periods
        self.statevectors = {}

    def __call__(self, P):
        '''
        Clamps state variable at reset value.
        '''
        # if we haven't computed the integer period for this group yet.
        # do so now
        if id(P) in self._periods:
            period = self._periods[id(P)]
        else:
            period = int(self.period / P.clock.dt) + 1
            self._periods[id(P)] = period
        V = self.statevectors.get(id(P), None)
        if V is None:
            V = P.state_(self.state)
            self.statevectors[id(P)] = V
        neuronindices = P.LS[0:period]
        if P._variable_refractory_time:
            neuronindices = neuronindices[P._next_allowed_spiketime[neuronindices] > (P.clock._t - P.clock._dt * 0.25)]
        V[neuronindices] = self.resetvalue

    def __repr__(self):
        return 'Refractory period, ' + str(self.period)


class SimpleCustomRefractoriness(Refractoriness):
    '''
    Holds the state variable at the custom reset value for a fixed time after a spike.
    
    **Initialised as:** ::
    
        SimpleCustomRefractoriness(resetfunc[,period=5*ms[,state=0]])
    
    with arguments:
    
    ``resetfun``
        The custom reset function ``resetfun(P, spikes)`` for ``P`` a
        :class:`NeuronGroup` and ``spikes`` a list of neurons that
        fired spikes.
    ``period``
        The length of time to hold at the reset value.
    ``state``
        The name or number of the state variable to reset and hold,
        it is your responsibility to check that this corresponds to
        the custom reset function.
    
    The assumption is that ``resetfun(P, spikes)`` will reset the state
    variable ``state`` on the group ``P`` for the spikes with indices
    ``spikes``. The values assigned by the custom reset function are
    stored by this object, and they are clamped at these values for
    ``period``. This object does not introduce refractoriness for more
    than the one specified variable ``state`` or for spike indices
    other than those in the variable ``spikes`` passed to the custom
    reset function.
    '''

    @check_units(period=second)
    def __init__(self, resetfun, period=5 * msecond, state=0):
        self.period = period
        self.resetfun = resetfun
        self.state = state
        self._periods = {} # a dictionary mapping group IDs to periods
        self.statevectors = {}
        self.lastresetvalues = {}

    def __call__(self, P):
        '''
        Clamps state variable at reset value.
        '''
        # if we haven't computed the integer period for this group yet.
        # do so now
        if id(P) in self._periods:
            period = self._periods[id(P)]
        else:
            period = int(self.period / P.clock.dt) + 1
            self._periods[id(P)] = period
        V = self.statevectors.get(id(P), None)
        if V is None:
            V = P.state_(self.state)
            self.statevectors[id(P)] = V
        LRV = self.lastresetvalues.get(id(P), None)
        if LRV is None:
            LRV = zeros(len(V))
            self.lastresetvalues[id(P)] = LRV
        lastspikes = P.LS.lastspikes()
        self.resetfun(P, lastspikes)             # call custom reset function 
        LRV[lastspikes] = V[lastspikes]         # store a copy of the custom resetted values
        clampedindices = P.LS[0:period]
        V[clampedindices] = LRV[clampedindices] # clamp at custom resetted values

    def __repr__(self):
        return 'Custom refractory period, ' + str(self.period)


class CustomRefractoriness(Refractoriness):
    '''
    Holds the state variable at the custom reset value for a fixed time after a spike.
    
    **Initialised as:** ::
    
        CustomRefractoriness(resetfunc[,period=5*ms[,refracfunc=resetfunc]])
    
    with arguments:
    
    ``resetfunc``
        The custom reset function ``resetfunc(P, spikes)`` for ``P`` a
        :class:`NeuronGroup` and ``spikes`` a list of neurons that
        fired spikes.
    ``refracfunc``
        The custom refractoriness function ``refracfunc(P, indices)`` for ``P`` a
        :class:`NeuronGroup` and ``indices`` a list of neurons that are in
        their refractory periods. In some cases, you can choose not to specify this,
        and it will use the reset function.
    ``period``
        The length of time to hold at the reset value.    
    '''

    @check_units(period=second)
    def __init__(self, resetfun, period=5 * msecond, refracfunc=None):
        self.period = period
        self.resetfun = resetfun
        if refracfunc is None:
            refracfunc = resetfun
        self.refracfunc = refracfunc
        self._periods = {} # a dictionary mapping group IDs to periods

    def __call__(self, P):
        '''
        Clamps state variable at reset value.
        '''
        # if we haven't computed the integer period for this group yet.
        # do so now
        if id(P) in self._periods:
            period = self._periods[id(P)]
        else:
            period = int(self.period / P.clock.dt) + 1
            self._periods[id(P)] = period
        lastspikes = P.LS.lastspikes()
        self.resetfun(P, lastspikes)             # call custom reset function
        clampedindices = P.LS[0:period]
        self.refracfunc(P, clampedindices)

    def __repr__(self):
        return 'Custom refractory period, ' + str(self.period)


class NoReset(Reset):
    '''
    Absence of reset mechanism.
    
    **Initialised as:** ::
    
        NoReset()
    '''
    def __init__(self):
        pass

    def __call__(self, P):
        pass

    def __repr__(self):
        return 'No reset'
