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
Threshold mechanisms
'''

__all__ = ['Threshold', 'FunThreshold', 'VariableThreshold', 'NoThreshold',
          'EmpiricalThreshold', 'SimpleFunThreshold', 'PoissonThreshold',
          'HomogeneousPoissonThreshold', 'StringThreshold']

from numpy import where, array, zeros, Inf
from units import *
from itertools import count
from clock import guess_clock, get_default_clock, reinit_default_clock
from random import sample # Python standard random module (sample is different)
from scipy import random
from numpy import clip
import bisect
from scipy import weave
from globalprefs import *
import warnings
from utils.approximatecomparisons import is_approx_equal
from log import *
import inspect
from inspection import *
import re
import numpy
CThreshold = PythonThreshold = None

def select_threshold(expr, eqs, level=0):
    '''
    Automatically selects the appropriate Threshold object from a string.
    
    Matches the following patterns:
    
    var_name > or >= const : Threshold
    var_name > or >= var_name : VariableThreshold
    others : StringThreshold
    '''
    global CThreshold, PythonThreshold
    use_codegen = get_global_preference('usecodegen') and get_global_preference('usecodegenthreshold')
    use_weave = get_global_preference('useweave') and get_global_preference('usecodegenweave')
    if use_codegen:
        if CThreshold is None:
            from experimental.codegen.threshold import CThreshold, PythonThreshold
        if use_weave:
            log_warn('brian.threshold', 'Using codegen CThreshold')
            return CThreshold(expr, level=level + 1)
        else:
            log_warn('brian.threshold', 'Using codegen PythonThreshold')
            return PythonThreshold(expr, level=level + 1)
    # plan:
    # - see if it matches A > B or A >= B, if not select StringThreshold
    # - check if A, B both match diffeq variable names, and if so
    #   select VariableThreshold
    # - check that A is a variable name, if not select StringThreshold
    # - extract all the identifiers from B, and if none of them are
    #   callable, assume it is a constant, try to eval it and then use
    #   Threshold. If not, or if eval fails, use StringThreshold.
    # This misses the case of e.g. V>10*mV*exp(1) because exp will be
    # callable, but in general a callable means that it could be
    # non-constant.
    expr = expr.strip()
    eqs.prepare()
    ns = namespace(expr, level=level + 1)
    s = re.search(r'^\s*(\w+)\s*>=?(.+)', expr)
    if not s:
        return StringThreshold(expr, level=level + 1)
    A = s.group(1)
    B = s.group(2).strip()
    if A not in eqs._diffeq_names:
        return StringThreshold(expr, level=level + 1)
    if B in eqs._diffeq_names:
        return VariableThreshold(B, A)
    try:
        vars = get_identifiers(B)
    except SyntaxError:
        return StringThreshold(expr, level=level + 1)
    all_vars = eqs._eq_names + eqs._diffeq_names + eqs._alias.keys() + ['t']
    for v in vars:
        if v not in ns or v in all_vars or callable(ns[v]):
            return StringThreshold(expr, level=level + 1)
    try:
        val = eval(B, ns)
    except:
        return StringThreshold(expr, level=level + 1)
    return Threshold(val, A)


class Threshold(object):
    '''
    All neurons with a specified state variable above a fixed value fire a spike.
    
    **Initialised as:** ::
    
        Threshold([threshold=1*mV[,state=0])
    
    with arguments:
    
    ``threshold``
        The value above which a neuron will fire.
    ``state``
        The state variable which is checked.
    
    **Compilation**
    
    Note that if the global variable ``useweave`` is set to ``True``
    then this function will use a ``C++`` accelerated version which
    runs approximately 3x faster.
    '''

    def __init__(self, threshold=1 * mvolt, state=0):
        self.threshold = threshold
        self.state = state
        self._useaccel = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
        self._extra_compile_args = ['-O3']
        if self._cpp_compiler == 'gcc':
            self._extra_compile_args += get_global_preference('gcc_options') # ['-march=native', '-ffast-math']

    def __call__(self, P):
        '''
        Checks the threshold condition and returns spike times.
        P is the neuron group.

        Note the accelerated version runs 3x faster.
        '''
        if self._useaccel:
            spikes = P._spikesarray
            V = P.state_(self.state)
            Vt = float(self.threshold)
            N = int(len(P))
            code = """
                    int numspikes=0;
                    for(int i=0;i<N;i++)
                        if(V(i)>Vt)
                            spikes(numspikes++) = i;
                    return_val = numspikes;
                    """
            # WEAVE NOTE: set the environment variable USER if your username has a space
            # in it, say set USER=DanGoodman if your username is Dan Goodman, this is
            # because weave uses this to create file names, but doesn't correctly send these
            # values to the compiler, causing problems.
            numspikes = weave.inline(code, ['spikes', 'V', 'Vt', 'N'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=self._extra_compile_args)
            # WEAVE NOTE: setting verbose=True in the weave.inline function may help in
            # finding errors.
            return spikes[0:numspikes]
        else:
            return ((P.state_(self.state) > self.threshold).nonzero())[0]

    def __repr__(self):
        return 'Threshold mechanism with value=' + str(self.threshold) + " acting on state " + str(self.state)


class StringThreshold(Threshold):
    '''
    A threshold specified by a string expression.
    
    Initialised with arguments:
    
    ``expr``
        The expression used to test whether a neuron has fired a spike.
        Should be a single statement that returns a value. For example,
        ``'V>50*mV'`` or ``'V>Vt'``.
    ``level``
        How many levels up in the calling sequence to look for
        names in the namespace. Usually 0 for user code.
    '''
    def __init__(self, expr, level=0):
        self._namespace, unknowns = namespace(expr, level=level + 1, return_unknowns=True)
        self._vars = unknowns
        self._expr = expr
        self._code = compile(expr, "StringThreshold", "eval")
        class Replacer(object):
            def __init__(self, func, n):
                self.n = n
                self.func = func
            def __call__(self):
                return self.func(self.n)
        self._Replacer = Replacer

    def __call__(self, P):
        for var in self._vars:
            self._namespace[var] = P.state(var)
        self._namespace['rand'] = self._Replacer(numpy.random.rand, len(P))
        self._namespace['randn'] = self._Replacer(numpy.random.randn, len(P))
        return eval(self._code, self._namespace).nonzero()[0]

    def __repr__(self):
        return "String threshold"


class NoThreshold(Threshold):
    '''
    No thresholding mechanism.
    
    **Initialised as:** ::
    
        NoThreshold()
    '''
    def __init__(self):
        pass

    def __call__(self, P):
        return []

    def __repr__(self):
        return "No Threshold"


class FunThreshold(Threshold):
    '''
    Threshold mechanism with a user-specified function.
    
    **Initialised as:** ::
    
        FunThreshold(thresholdfun)
    
    where ``thresholdfun`` is a function with one argument,
    the 2d state value array, where each row is an array of
    values for one state, of length N for N the number of
    neurons in the group. For efficiency, data are numpy
    arrays and there is no unit checking.
    
    Note: if you only need to consider one state variable,
    use the :class:`SimpleFunThreshold` object instead.
    '''
    def __init__(self, thresholdfun):
        self.thresholdfun = thresholdfun # Threshold function

    def __call__(self, P):
        '''
        Checks the threshold condition and returns spike times.
        P is the neuron group.
        '''
        spikes = (self.thresholdfun(*P._S).nonzero())[0]
        return spikes

    def __repr__(self):
        return 'Functional threshold mechanism'


class SimpleFunThreshold(FunThreshold):
    '''
    Threshold mechanism with a user-specified function.
    
    **Initialised as:** ::
    
        FunThreshold(thresholdfun[,state=0])
    
    with arguments:
    
    ``thresholdfun``
        A function with one argument, the array of values for
        the specified state variable. For efficiency, this is
        a numpy array, and there is no unit checking.
    ``state``
        The name or number of the state variable to pass to
        the threshold function.
    
    **Sample usage:** ::
    
        FunThreshold(lambda V:V>=Vt,state='V')
    '''
    def __init__(self, thresholdfun, state=0):
        self.thresholdfun = thresholdfun # Threshold function
        self.state = state

    def __call__(self, P):
        '''
        Checks the threshold condition and returns spike times.
        P is the neuron group.
        '''
        spikes = (self.thresholdfun(P.state_(self.state)).nonzero())[0]
        #P.LS[spikes]=P.clock.t # Time of last spike (this line should be general)
        return spikes


class VariableThreshold(Threshold):
    '''
    Threshold mechanism where one state variable is compared to another.
    
    **Initialised as:** ::
    
        VariableThreshold([threshold_state=1[,state=0]])
        
    with arguments:
    
    ``threshold_state``
        The state holding the lower bound for spiking.
    ``state``
        The state that is checked.
    
    If ``x`` is the value of state variable ``threshold_state`` on neuron
    ``i`` and ``y`` is the value of state variable ``state`` on neuron
    ``i`` then neuron ``i`` will fire if ``y>x``.
    
    Typically, using this class is more time efficient than writing
    a custom thresholding operation.
    
    **Compilation**
    
    Note that if the global variable ``useweave`` is set to ``True``
    then this function will use a ``C++`` accelerated version.
    '''
    def __init__(self, threshold_state=1, state=0):
        self.threshold_state = threshold_state # State variable representing the threshold
        self.state = state
        self._useaccel = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
        self._extra_compile_args = ['-O3']
        if self._cpp_compiler == 'gcc':
            self._extra_compile_args += get_global_preference('gcc_options') # ['-march=native', '-ffast-math']

    def __call__(self, P):
        '''
        Checks the threshold condition, resets and returns spike times.
        P is the neuron group.
        '''
        if self._useaccel:
            spikes = P._spikesarray
            V = P.state_(self.state)
            Vt = P.state_(self.threshold_state)
            N = int(len(P))
            code = """
                    int numspikes=0;
                    for(int i=0;i<N;i++)
                        if(V(i)>Vt(i))
                            spikes(numspikes++) = i;
                    return_val = numspikes;
                    """
            numspikes = weave.inline(code, ['spikes', 'V', 'Vt', 'N'], \
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=self._extra_compile_args)
            return spikes[0:numspikes]
        else:
            return ((P.state_(self.state) > P.state_(self.threshold_state)).nonzero())[0]

    def __repr__(self):
        return 'Variable threshold mechanism'


class EmpiricalThreshold(Threshold):
    '''
    Empirical threshold, e.g. for Hodgkin-Huxley models.
    
    In empirical models such as the Hodgkin-Huxley method, after a spike
    neurons are not instantaneously reset, but reset themselves
    as part of the dynamical equations defining their behaviour. This class
    can be used to model that. It is a simple threshold mechanism that
    checks e.g. ``V>=Vt`` but it only does so for neurons that haven't
    recently fired (giving the dynamical equations time to reset
    the values naturally). It should be used in conjunction with the
    :class:`NoReset` object.
    
    **Initialised as:** ::
    
        EmpiricalThreshold([threshold=1*mV[,refractory=1*ms[,state=0[,clock]]]])

    with arguments:
    
    ``threshold``
        The lower bound for the state variable to induce a spike.
    ``refractory``
        The time to wait after a spike before checking for spikes again.
    ``state``
        The name or number of the state variable to check.
    ``clock``
        If this object is being used for a :class:`NeuronGroup` which doesn't
        use the default clock, you need to specify its clock here.
    '''
    @check_units(refractory=second)
    def __init__(self, threshold=1 * mvolt, refractory=1 * msecond, state=0, clock=None):
        self.threshold = threshold # Threshold value
        self.state = state
        clock = guess_clock(clock)
        self.refractory = int(refractory / clock.dt)
        # this assumes that if the state stays over the threshold, and say
        # refractory=5ms the user wants spiking at 0ms 5ms 10ms 15ms etc.
        if is_approx_equal(self.refractory * clock.dt, refractory) and self.refractory > 0:
            self.refractory -= 1

    def __call__(self, P):
        '''
        Checks the threshold condition, resets and returns spike times.
        P is the neuron group.
        '''
        #spikes=where((P._S[0,:]>self.Vt) & ((P.LS<P.clock.t-self.refractory) | (P.LS==P.clock.t)))[0]
        spikescond = P.state_(self.state) > self.threshold
        spikescond[P.LS[0:self.refractory]] = False
        return spikescond.nonzero()[0]
        #P.LS[spikes]=P.clock.t # Time of last spike (this line should be general)
        #return spikes

    def __repr__(self):
        return 'Empirical threshold with value=' + str(self.threshold) + " acting on state " + str(self.state)


class PoissonThreshold(Threshold):
    '''
    Poisson threshold: a spike is produced with some probability S[0]*dt,
    or S[state]*dt.
    '''
    # TODO: check the state has units in Hz
    def __init__(self, state=0):
        self.state = state

    def __call__(self, P):
        return (random.rand(len(P)) < P.state_(self.state)[:] * P.clock.dt).nonzero()[0]

    def __repr__(self):
        return 'Poisson threshold'


class HomogeneousPoissonThreshold(PoissonThreshold):
    '''
    Poisson threshold for spike trains with identical rates.
    The underlying NeuronGroup has only one state variable.
    N.B.: "homogeneous" is meant in the spatial (not temporal) sense,
    the rate may change in time.
    '''
    def __call__(self, P):
        # N.B.: is "float" necessary?
        # Other possibility to avoid sorting: use an exponential distribution
        n = random.poisson(float(len(P) * P.clock.dt * clip(P._S[self.state][0], 0, Inf))) # number of spikes
        if n > len(P):
            n = len(P)
            log_warn('brian.HomogeneousPoissonThreshold', 'HomogeneousPoissonThreshold cannot generate enough spikes.')
        spikes = sample(xrange(len(P)), n)
        spikes.sort() # necessary only for subgrouping
        return spikes
