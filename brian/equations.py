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
Differential equations for Brian models.
'''
#from scipy.weave import blitz
from operator import isSequenceType
import types
from units import *
from stdunits import *
from inspection import *
from scipy import exp
from scipy import weave
from globalprefs import *
import re
import inspect
import optimiser
import warnings
import uuid
import numpy
from numpy import zeros, ones
import numpy
from log import *
from optimiser import *
from scipy import optimize
import unitsafefunctions
import copy
try:
    import sympy
    use_sympy = True
except:
    warnings.warn('sympy not installed')
    use_sympy = False

__all__ = ['Equations', 'unique_id']

# TODO: write interface for equations.py

def unique_id():
    """
    Returns a unique name (e.g. for internal hidden variables).
    """
    return '_' + str(uuid.uuid1().int)


class Equations(object):
    """Container that stores equations from which models can be created
    
    Initialised as::
    
        Equations(expr[,level=0[,keywords...]])
    
    with arguments:
    
    ``expr``
        An expression, which can each be a string representing equations,
        an :class:`Equations` objects, or a list of strings and :class:`Equations` objects.
        See below for details of the string format.
    ``level``
        Indicates how many levels back in the stack the namespace for string
        equations is found, so that e.g. ``level=0`` looks in the
        namespace of the function where the :class:`Equations` object was created,
        ``level=1`` would look in the namespace of the function that called the
        function where the :class:`Equations` object was created, etc.
        Normally you can just leave this out.
    ``keywords``
        Any sequence of keyword pairs ``key=value`` where the string ``key``
        in the string equations will be replaced with ``value`` which can
        be either a string, value or ``None``, in the latter case a unique
        name will be generated automatically (but it won't be pretty).
    
    Systems of equations can be defined by passing lists of :class:`Equations` to a
    new :class:`Equations` object, or by adding :class:`Equations` objects together (the usage
    is similar to that of a Python ``list``).
    
    **String equations**
    
    String equations can be of any of the following forms:
    
    (1) ``dx/dt = f : unit`` (differential equation)
    (2) ``x = f : unit`` (equation)
    (3) ``x = y`` (alias)
    (4) ``x : unit`` (parameter)
    
    Here each of ``x`` and ``y`` can be any valid Python variable name,
    ``f`` can be any valid Python expression, and ``unit`` should be the
    unit of the corresponding ``x``. You can also include multi-line
    expressions by appending a ``\`` character at the end of each line
    which is continued on the next line (following the Python standard),
    or comments by including a ``#`` symbol.
    
    These forms mean:
    
    *Differential equation*
        A differential equation with variable ``x`` which has physical
        units ``unit``. The variable ``x`` will become one of the state
        variables of the model.
    *Equation*
        An equation defining the meaning of ``x`` can be used for building
        systems of complicated differential equations.
    *Alias*
        The variable ``x`` becomes equivalent to the variable ``y``, useful
        for connecting two separate systems of equations together.
    *Parameter*
        The variable ``x`` will have physical units ``unit`` and will be
        one of the state variables of the model (but will not evolve
        dynamically, instead it should be set by the user).
        
    .. index::
        single: xi
        pair: xi; noise
        single: white noise
        single: gaussian noise
        single: noise
        single: noise; gaussian
        single: noise; white
    
    **Noise**
        
    String equations can also use the reserved term ``xi`` for a
    Gaussian white noise with mean 0 and variance 1.
    
    **Example usage** ::

        eqs=Equations('''
        dv/dt=(u-v)/tau : volt
        u=3*v : volt
        w=v
        ''')
    
    **Details**
    
    For more details, see :ref:`moreonequations` in the user manual.
    """
    def __init__(self, expr='', level=0, **kwds):
        # Empty object
        self._Vm = None # name of variable with membrane potential
        self._eq_names = [] # equations names
        self._diffeq_names = [] # differential equations names
        self._diffeq_names_nonzero = [] # differential equations names
        self._function = {} # dictionary of functions
        self._string = {} # dictionary of strings (defining the functions)
        self._namespace = {} # dictionary of namespaces for the strings (globals,locals)
        self._alias = {} # aliases (mapping name1 -> name2)
        self._units = {'t':second} # dictionary of units
        self._dependencies = {} # dictionary of dependencies (on static equations)
        self._useweave = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
        self._extra_compile_args = ['-O3']
        if self._cpp_compiler == 'gcc':
            self._extra_compile_args += get_global_preference('gcc_options') # ['-march=native', '-ffast-math']
        self._frozen = False # True if all units and parameters are gone
        self._prepared = False

        if not isinstance(expr, str): # assume it is a sequence of Equations objects
            for eqs in expr:
                if not isinstance(eqs, Equations):
                    eqs = Equations(eqs, level=level + 1)
                self += eqs
        elif expr != '':
            # Check keyword arguments
            param_dict = {}
            for name, value in kwds.iteritems():
                if value is None: # name is not important: choose unique name
                    value = unique_id()
                if isinstance(value, str): # variable name substitution
                    expr = re.sub('\\b' + name + '\\b', value, expr)
                    expr = re.sub('\\bd' + name + '\\b', 'd' + value, expr) # derivative
                else:
                    param_dict[name] = value

            if kwds == {}: # weird: changed from param_dict on 18/06/08
                self.parse_string_equations(expr, level=level + 1)
            else:
                self.parse_string_equations(expr, namespace=param_dict, level=level + 1)

    """
    -----------------------------------------------------------------------
    PARSING AND BUILDING NAMESPACES
    -----------------------------------------------------------------------
    """

    def parse_string_equations(self, eqns, level=1, namespace=None):
        """
        Parses a string defining equations and builds an Equations object.
        Uses the namespace in the given level of the stack.
        """
        diffeq_pattern = re.compile('\s*d(\w+)\s*/\s*dt\s*=\s*(.+?)\s*:\s*(.*)')
        eq_pattern = re.compile('\s*(\w+)\s*=\s*(.+?)\s*:\s*(.*)')
        alias_pattern = re.compile('\s*(\w+)\s*=\s*(\w+)\s*$')
        param_pattern = re.compile('\s*(\w+)\s*:\s*(.*)')
        empty_pattern = re.compile('\s*$')
        patterns = [diffeq_pattern, eq_pattern, alias_pattern, param_pattern, empty_pattern]
        # Merge multi-line statements
        eqns = re.sub('\\\s*?\n', ' ', eqns)

        # Namespace of the functions
        ns_global, ns_local = namespace, namespace
        if namespace is None:
            frame = inspect.stack()[level + 1][0]
            ns_global, ns_local = frame.f_globals, frame.f_locals
            #print frame.f_code.co_filename #useful for debugging which file the namespace came from

        for line in eqns.splitlines():
            line = re.sub('#.*', '', line) # remove comments
            result = None
            for pattern in patterns:
                result = pattern.match(line)
                if result:
                    break
            if result == None:
                raise TypeError, "Invalid equation string: " + line
            if pattern == eq_pattern:
                name, eq, unit = result.groups()
                self.add_eq(name, eq, unit, ns_global, ns_local)
            elif pattern == diffeq_pattern:
                name, eq, unit = result.groups()
                self.add_diffeq(name, eq, unit, ns_global, ns_local)
            elif pattern == alias_pattern:
                name1, name2 = result.groups()
                self.add_alias(name1, name2)
            elif pattern == param_pattern:
                name, unit = result.groups()
                self.add_param(name, unit, ns_global, ns_local)

    def add_eq(self, name, eq, unit, global_namespace={}, local_namespace={}):
        """
        Inserts an equation.
        name = variable name
        eq = string definition
        unit = unit of the variable (possibly a string)
        *_namespace = namespaces associated to the string
        """
        # Find external objects
        vars = list(get_identifiers(eq))
        if type(unit) == types.StringType:
            vars.extend(list(get_identifiers(unit)))
        self._namespace[name] = {}
        for var in vars:
            if var in local_namespace: #local
                self._namespace[name][var] = local_namespace[var]
            elif var in global_namespace: #global
                self._namespace[name][var] = global_namespace[var]
            elif var in globals(): # typically units
                self._namespace[name][var] = globals()[var]

        self._eq_names.append(name)
        if type(unit) == types.StringType:
            self._units[name] = eval(unit, self._namespace[name].copy())
        else:
            self._units[name] = unit
        self._string[name] = eq

    def add_diffeq(self, name, eq, unit, global_namespace={}, local_namespace={}, nonzero=True):
        """
        Inserts a differential equation.
        name = variable name
        eq = string definition
        unit = unit of the variable (possibly a string)
        *_namespace = namespaces associated to the string
        nonzero = False if dx/dt=0 (parameter)
        """
        # Find external objects
        vars = list(get_identifiers(eq))
        if type(unit) == types.StringType:
            vars.extend(list(get_identifiers(unit)))
        self._namespace[name] = {}
        for var in vars:
            if var in local_namespace: #local
                self._namespace[name][var] = local_namespace[var]
            elif var in global_namespace: #global
                self._namespace[name][var] = global_namespace[var]
            elif var in globals(): # typically units
                self._namespace[name][var] = globals()[var]

        self._diffeq_names.append(name)
        if type(unit) == types.StringType:
            self._units[name] = eval(unit, self._namespace[name].copy())
        else:
            self._units[name] = unit
        self._string[name] = eq
        if nonzero:
            self._diffeq_names_nonzero.append(name)

    def add_alias(self, name1, name2):
        """
        Inserts an alias.
        name1 = new name
        name2 = old name
        """
        self._alias[name1] = name2
        # TODO: what if name2 is not defined yet?
        self.add_eq(name1, name2, self._units[name2])

    def add_param(self, name, unit, global_namespace={}, local_namespace={}):
        """
        Inserts a parameter.
        name = variable name
        eq = string definition
        unit = unit of the variable (possibly a string)
        *_namespace = namespaces associated to the string
        """
        if isinstance(unit, Quantity):
            unit = scalar_representation(unit)
        self.add_diffeq(name, '0*' + unit + '/second', unit, global_namespace, local_namespace, nonzero=False)

    """
    -----------------------------------------------------------------------
    FINALISATION
    -----------------------------------------------------------------------
    """

    def prepare(self, check_units=True):
        '''
        Do a number of checks (units) and preparation of the object.
        '''
        if self._prepared:
            return
        # Let Vm be the first differential equation
        vm_name = self.get_Vm()
        if vm_name:
            # TODO: INFO logging
            i = self._diffeq_names.index(vm_name)
            self._diffeq_names[0], self._diffeq_names[i] = self._diffeq_names[i], self._diffeq_names[0]
        else:
            pass
            # TODO: WARNING log that a potential problem has occurred here?

        # Clean namespace (avoids conflicts between variables and external variables)
        self.clean_namespace()
        # Compile strings to functions
        self.compile_functions()
        # Check units
        if check_units: self.check_units()
        # Set the update order of (static) variables
        self.set_eq_order()
        # Replace static variables by their value in differential equations
        self.substitute_eq()
        self.compile_functions()

        # Check free variables
        free_vars = self.free_variables()
        if free_vars != []:
            log_info('brian.equations', 'Free variables: ' + str(free_vars))

        self._prepared = True

    def get_Vm(self):
        '''
        Finds the variable that is most likely to be the
        membrane potential.
        '''
        if self._Vm:
            return self._Vm
        vm_names = ['v', 'V', 'vm', 'Vm']
        guesses = [var for var in self._diffeq_names if var in vm_names]
        if len(guesses) == 1: # Unambiguous
            return guesses[0]
        else: # Ambiguous or not found
            return None

    def clean_namespace(self):
        '''
        Removes all variable names from namespaces
        '''
        all_variables = self._eq_names + self._diffeq_names + self._alias.keys() + ['t']
        for name in self._namespace:
            for var in all_variables:
                if var in self._namespace[name]:
                    log_warn('brian.equations', 'Equation variable ' + var + ' also exists in the namespace')
                    del self._namespace[name][var]

    def compile_functions(self, freeze=False):
        """
        Compile all functions defined as strings.
        If freeze is True, all external parameters and units are replaced by their value.
        ALL FUNCTIONS MUST HAVE STRINGS.
        """
        all_variables = self._eq_names + self._diffeq_names + self._alias.keys() + ['t']
        # Check if freezable
        freeze = freeze and all([optimiser.freeze(expr, all_variables, self._namespace[name])\
                               for name, expr in self._string.iteritems()])
        self._frozen = freeze

        # Compile strings to functions
        for name, expr in self._string.iteritems():
            namespace = self._namespace[name] # name space of the function
            # Find variables
            vars = [var for var in get_identifiers(expr) if var in all_variables]
            if freeze:
                expr = optimiser.freeze(expr, all_variables, namespace)
                #self._string[name]=expr # should we?
                #namespace={}
            s = "lambda " + ','.join(vars) + ":" + expr
            self._function[name] = eval(s, namespace)

    def check_units(self):
        '''
        Checks the units of the differential equations, using
        the units of x.
        dx_i/dt must have units of x_i / time.
        '''
        self.set_eq_order()
        # Better: replace xi in the string, or in the namespace
        try:
            for var in self._eq_names:
                f = self._function[var]
                old_func_globals = copy.copy(f.func_globals)
                f.func_globals['xi'] = 0 * second ** -.5 # Noise
                f.func_globals.update(namespace_replace_quantity_with_pure(f.func_globals))
                units = namespace_replace_quantity_with_pure(self._units)
                self.apply(var, units) + self._units[var] # Check that the two terms have the same dimension
                f.func_globals.update(old_func_globals)
            for var in self._diffeq_names:
                f = self._function[var]
                old_func_globals = copy.copy(f.func_globals)
                f.func_globals['xi'] = 0 * second ** -.5 # Noise
                f.func_globals.update(namespace_replace_quantity_with_pure(f.func_globals))
                units = namespace_replace_quantity_with_pure(self._units)
                self.apply(var, units) + (self._units[var] / second) # Check that the two terms have the same dimension
                f.func_globals.update(old_func_globals)
        except DimensionMismatchError, inst:
            raise DimensionMismatchError("The differential equation of " + var + " is not homogeneous", *inst._dims)
        except:
            warnings.warn("Unexpected exception in checking units of " + var)
            raise

    def set_eq_order(self):
        '''
        Computes the internal depency graph of static variables
        and deduces the update order.
        Sets the list of dependencies of dynamic variables on static variables.
        This is called by check_units()
        '''
        if len(self._eq_names) > 0:
            # Internal dependency dictionary
            dependency = {}
            for key in self._eq_names:
                f = self._function[key]
                dependency[key] = [var for var in f.func_code.co_varnames if var in self._eq_names]

            # Sets the order
            staticvars_list = []
            no_dep = None
            while (len(staticvars_list) < len(self._eq_names)) and (no_dep != []):
                no_dep = [key for key, value in dependency.iteritems() if value == []]
                staticvars_list += no_dep
                # Clear dependency list
                for key in no_dep:
                    del dependency[key]
                for key, value in dependency.iteritems():
                    dependency[key] = [var for var in value if not(var in staticvars_list)]

            if no_dep == []: # The dependency graph has cycles!
                raise ReferenceError, "The static variables are referring to each other"
        else:
            staticvars_list = []

        # Calculate dependencies on static variables
        self._dependencies = {}
        for key in staticvars_list:
            self._dependencies[key] = []
            for var in self._function[key].func_code.co_varnames:
                if var in self._eq_names:
                    self._dependencies[key] += [var] + self._dependencies[var]
        for key in self._diffeq_names:
            f = self._function[key]
            self._dependencies[key] = []
            for var in f.func_code.co_varnames:
                if var in self._eq_names:
                    self._dependencies[key] += [var] + self._dependencies[var]

        # Sort the dependency lists
        for key in self._dependencies:
            staticdep = [(staticvars_list.index(var), var) for var in self._dependencies[key]]
            staticdep.sort()
            self._dependencies[key] = [x[1] for x in staticdep]

        # Update _eq
        self._eq_names = staticvars_list

    def substitute_eq(self, name=None):
        """
        Replaces the static variable 'name' by its value in differential
        equations.
        If None: substitute all static variables.
        """
        if name is None:
            for var in self._eq_names[-1::-1]: # reverse order
                self.substitute_eq(var)
        else:
            self.add_prefix_namespace(name)
            #print name
            for var in self._diffeq_names_nonzero:
                # String
                self._string[var] = re.sub("\\b" + name + "\\b", '(' + self._string[name] + ')', self._string[var])
                # Namespace
                self._namespace[var].update(self._namespace[name])
            #print self

    def add_prefix_namespace(self, name):
        """
        Make the variables in the namespace associated to variable name
        specific to that variable by inserting the prefix name_.
        """
        vars = self._namespace[name].keys()
        untransformed_funcs = set(getattr(unitsafefunctions, v) for v in unitsafefunctions.quantity_versions)
        untransformed_funcs.update(set([numpy.clip]))
        for var in vars:
            v = self._namespace[name][var]
            addprefix = True
            if isinstance(v, numpy.ufunc):
                addprefix = False
            try:
                if v in untransformed_funcs:
                    addprefix = False
            except TypeError: #unhashable types
                pass
            if addprefix:
                # String
                self._string[name] = re.sub("\\b" + var + "\\b", name + '_' + var, self._string[name])
                # Namespace
                self._namespace[name][name + '_' + var] = self._namespace[name][var]
                del self._namespace[name][var]

    def free_variables(self):
        """
        Returns the list of free variables (i.e., which are not defined within the
        equation string).
        """
        all_variables = self._eq_names + self._diffeq_names + self._alias.keys() + ['t']
        free_vars = []
        for expr in self._string.itervalues():
            free_vars += [name for name in get_identifiers(expr) if name not in all_variables]
        return list(set(free_vars))

    """
    -----------------------------------------------------------------------
    CALCULATING RHS OF DIFF EQUATIONS
    -----------------------------------------------------------------------
    """
    def apply(self, state, vardict):
        '''
        Calculates self._function[state] with arguments in vardict and
        static variables. The dictionary is filled with the required
        static variables.
        '''
        f = self._function[state]
        # Calculate static variables
        for var in self._dependencies[state]:
            # could add something like: if var not in vardict: this would allow you to override the dependencies if you wanted to - worth doing?
            vardict[var] = call_with_dict(self._function[var], vardict)
        return f(*[vardict[var] for var in f.func_code.co_varnames])

    """
    -----------------------------------------------------------------------
    EQUATION INSPECTION
    -----------------------------------------------------------------------
    """
    def is_stochastic(self, var=None):
        '''
        Returns True if the equation for var is stochastic,
        or if all equations are stochastic (var=None).
        '''
        if var:
            return 'xi' in get_identifiers(self._string[var])
        else:
            return any([self.is_stochastic(name) for name in self._diffeq_names_nonzero])

    def is_time_dependent(self, var=None):
        '''
        Returns True if the equation for var is time dependent,
        or if all equations are time dependent (var=None).
        '''
        if var:
            return 't' in get_identifiers(self._string[var])
        else:
            return any([self.is_time_dependent(name) for name in self._diffeq_names_nonzero])
#            return any([self.is_time_dependent(name) for name in self._diffeq_names_nonzero+self._eq_names])

    def is_linear(self):
        '''
        Returns True if all equations are linear.
        If the equations are time dependent, then returns False.
        If the equations depend on external functions, then returns False.
        '''
        if self.is_time_dependent():
            return False

        for f in self._namespace.iterkeys():
            if any([type(key) == types.FunctionType for key in self._namespace[f].itervalues()]):
                return False
        return all([is_affine(f) for f in self._function.itervalues()])

    def is_conditionally_linear(self):
        '''
        Returns True if the differential equations are linear with respect to the
        state variable.
        '''
        # Equations have to be prepared for it to work.
        for var in self._diffeq_names:
            S = self._units.copy()
            S[var] = AffineFunction()
            try:
                self.apply(var, S)
            except:
                return False
        return True

    """
    -----------------------------------------------------------------------
    NUMERICAL INTEGRATION (to be replaced by code generation)
    -----------------------------------------------------------------------
    """

    def forward_euler(self, S, dt):
        '''
        Updates the value of the state variables in dictionary S
        with the forward Euler algorithm over step dt.
        '''
        # Calculate all static variables (or do that after?)
        #for var in self._eq_names:
        #    S[var]=call_with_dict(self._function[var],S)
        # Calculate derivatives
        buffer = {}
        for varname in self._diffeq_names_nonzero:
            f = self._function[varname]
            buffer[varname] = f(*[S[var] for var in f.func_code.co_varnames])
        # Update variables
        for var in self._diffeq_names_nonzero:
            S[var] += dt * buffer[var]

    def forward_euler_code_string(self):
        '''
        Generates Python code for a forward Euler step.
        '''
        # TODO: check if it can really be frozen
        # TODO: change /a to *(1/a) with precalculation (use parser)
        all_variables = self._eq_names + self._diffeq_names + self._alias.keys() + ['t']
        # nonzero? insert dt?
        vars_tmp = [name + '__tmp' for name in self._diffeq_names]
        lines = ','.join(self._diffeq_names) + '=P._S\n'
        lines += ','.join(vars_tmp) + '=P._dS\n'
        for name in self._diffeq_names_nonzero:
            namespace = self._namespace[name]
            expr = optimiser.freeze(self._string[name], all_variables, namespace)
            lines += name + '__tmp[:]=' + expr + '\n'
        lines += 'P._S+=dt*P._dS\n'
        #print lines
        return lines
        # Return a function f(P) or a namespace (exec code in namespace)
        # 1st option: include directly in neurongroup._state_updater (good?)        

    def forward_euler_code(self):
        '''
        Generates Python code for a forward Euler step.
        '''
        # TODO: check if it can really be frozen
        # TODO: change /a to *(1/a) with precalculation (use parser)
        all_variables = self._eq_names + self._diffeq_names + self._alias.keys() + ['t']
        # nonzero? insert dt?
        vars_tmp = [name + '__tmp' for name in self._diffeq_names]
        lines = ','.join(self._diffeq_names) + '=P._S\n'
        lines += ','.join(vars_tmp) + '=P._dS\n'
        for name in self._diffeq_names_nonzero:
            namespace = self._namespace[name]
            expr = optimiser.freeze(self._string[name], all_variables, namespace)
            lines += name + '__tmp[:]=' + expr + '\n'
        lines += 'P._S+=dt*P._dS\n'
        #print lines
        return compile(lines, 'Euler update code', 'exec')
        # Return a function f(P) or a namespace (exec code in namespace)
        # 1st option: include directly in neurongroup._state_updater (good?)

    def Runge_Kutta2(self, S, dt):
        '''
        Updates the value of the state variables in dictionary S
        with the 2nd order Runge-Kutta algorithm over step dt (midpoint).
        '''
        # Calculate all static variables (or do that after?)
        #for var in self._eq_names:
        #    S[var]=call_with_dict(self._function[var],S)
        # Calculate derivatives
        buffer = {}
        S_half = S.copy()
        # Half a step
        for varname in self._diffeq_names_nonzero:
            f = self._function[varname]
            buffer[varname] = f(*[S[var] for var in f.func_code.co_varnames])
        # Update variables
        for var in self._diffeq_names_nonzero:
            S_half[var] = S[var] + .5 * dt * buffer[var]

        # Whole step
        for varname in self._diffeq_names_nonzero:
            f = self._function[varname]
            buffer[varname] = f(*[S_half[var] for var in f.func_code.co_varnames])
        # Update variables
        for var in self._diffeq_names_nonzero:
            S[var] += dt * buffer[var]

    def exponential_euler(self, S, dt):
        '''
        Updates the value of the state variables in dictionary S
        with an exponential Euler algorithm over step dt.
        Test with is_conditionally_linear first.
        Same as default integration method in Genesis.
        Close to the implicit Euler method in Neuron.
        '''
        # Calculate all static variables (BAD: INSERT IT BELOW)
        #for var in self._eq_names:
        #    S[var]=call_with_dict(self._function[var],S)
        n = len(S[self._diffeq_names_nonzero[0]])
        # Calculate the coefficients of the affine function
        Z = zeros(n)
        O = ones(n)
        A = {}
        B = {}
        for varname in self._diffeq_names_nonzero:
            f = self._function[varname]
            oldval = S[varname]
            S[varname] = Z
            B[varname] = f(*[S[var] for var in f.func_code.co_varnames]).copy() # important if compiled
            S[varname] = O
            A[varname] = f(*[S[var] for var in f.func_code.co_varnames]) - B[varname]
            B[varname] /= A[varname]
            S[varname] = oldval
        # Integrate
        for varname in self._diffeq_names_nonzero:
            f = self._function[varname]
            if self._useweave:
                Bx = B[varname]
                Ax = A[varname]
                Sx = S[varname]
                # Compilation with blitz: we need an approximation because exp is not understood
                #weave.blitz('Sx[:]=-Bx+(Sx+Bx)*(1.+Ax*dt*(1.+.5*Ax*dt))',check_size=0)
                code = """
                for(int k=0;k<n;k++)
                    Sx(k)=-Bx(k)+(Sx(k)+Bx(k))*exp(Ax(k)*dt);
                """
                weave.inline(code, ['n', 'Bx', 'Sx', 'Ax', 'dt'], \
                             compiler=self._cpp_compiler,
                             type_converters=weave.converters.blitz,
                             extra_compile_args=self._extra_compile_args)
            else:
                #S[varname][:]=-B[varname]+(S[varname]+B[varname])*exp(A[varname]*dt)
                # A little faster:
                S[varname] += B[varname]
                S[varname] *= exp(A[varname] * dt)
                S[varname] -= B[varname]

    def exponential_euler_code(self):
        '''
        Generates Python code for an exponential Euler step.
        Not efficient for the moment!
        '''
        all_variables = self._eq_names + self._diffeq_names + self._alias.keys() + ['t']
        vars_tmp = [name + '__tmp' for name in self._diffeq_names]
        lines = ','.join(self._diffeq_names) + '=P._S\n'
        lines += ','.join(vars_tmp) + '=P._dS\n'
        for name in self._diffeq_names:
            # Freeze
            namespace = self._namespace[name]
            expr = optimiser.freeze(self._string[name], all_variables, namespace)
            # Find a and b in dx/dt=a*x+b
            sym_expr = symbolic_eval(expr)
            if isinstance(sym_expr, float):
                lines += name + '__tmp[:]=' + name + '+(' + str(expr) + ')*dt\n'
            else:
                sym_expr = sym_expr.expand()
                sname = sympy.Symbol(name)
                terms = sympy.collect(sym_expr, name, evaluate=False)
                if sname ** 0 in terms:
                    b = terms[sname ** 0]
                else:
                    b = 0
                if sname in terms:
                    a = terms[sname]
                else:
                    a = 0
                lines += name + '__tmp[:]=' + str(-b / a + (sname + b / a) * sympy.exp(a * sympy.Symbol('dt'))) + '\n'
        lines += 'P._S[:]=P._dS'
        #print lines
        return compile(lines, 'Exponential Euler update code', 'exec')

    """
    -------------------
    COMBINING EQUATIONS
    -------------------
    """
    def __add__(self, other):
        '''
        Union of two sets of equations
        '''
        if not isinstance(other, Equations):
            other = Equations(other, level=1)
        result = self.__class__()
        result += self
        result += other
        return result
    __radd__ = __add__

    def __iadd__(self, other):
        if not isinstance(other, Equations):
            other = Equations(other, level=1)
        self._eq_names = list(set(self._eq_names + other._eq_names)) # what to do if same variables?
        self._diffeq_names = list(set(self._diffeq_names + other._diffeq_names))
        self._diffeq_names_nonzero = list(set(self._diffeq_names_nonzero + other._diffeq_names_nonzero))
        self._function = disjoint_dict_union(self._function, other._function)
        self._alias = disjoint_dict_union(self._alias, other._alias)
        self._string = disjoint_dict_union(self._string, other._string)
        self._namespace = disjoint_dict_union(self._namespace, other._namespace)
        # We do this to fix a bug where if you add two Equations together and
        # then create groups from them, the add_prefix_namespace step creates
        # names which can't be correctly resolved in the second NeuronGroup
        # created. This happens because although self._namespace is a new object,
        # self._namespace[var] is shared between the two objects.
        for var in self._namespace.keys():
            self._namespace[var] = copy.copy(self._namespace[var])
        try:
            self._units = disjoint_dict_union(self._units, other._units)
        except AttributeError:
            raise DimensionMismatchError("The two sets of equations do not have compatible units")
        return self

    """
    ---------------------------------
    OTHER METHODS (CALLED EXTERNALLY)
    ---------------------------------
    """
    def fixed_point(self, **kwd):
        '''
        Returns a fixed point of the differential equations
        as a dictionary. The keyword arguments give the (optional)
        initial point (default = 0).
        '''
        values = {}
        for name, value in self._units.iteritems():
            values[name] = 0 * value
        values.update(kwd)
        # Initial vector
        x0 = [values[name] for name in self._diffeq_names_nonzero]
        # Vector function
        def f(x):
            # Put the units back
            x = [xi * get_unit(x0i) for xi, x0i in zip(x, x0)]
            values.update(zip(self._diffeq_names_nonzero, x))
            return [self.apply(name, values) for name in self._diffeq_names_nonzero]
        xf, _, ier, _ = optimize.fsolve(f, x0, full_output=True)
        if ier:
            # Put the units back
            xf = [xfi * get_unit(x0i) for xfi, x0i in zip(xf, x0)]
            # Return a dictionary
            return dict(zip(self._diffeq_names_nonzero, xf))
        else: # Not found
            warnings.warn('Could not find a fixed point of the equations')
            return kwd

    def substitute(self, name1, name2):
        """
        Changes name1 to name2 (variable names).
        Note: I don't where this is called!
        """
        # Aliases
        if name1 in self._alias:
            self._alias[name2] = self._alias[name1]
            del self._alias[name1]
        # Units
        if name1 in self._units:
            self._units[name2] = self._units[name1]
            del self._units[name1]
        # Equations
        if name1 in self._eq_names:
            self._eq_names[self._eq_names.index(name1)] = name2
        # Differential equations
        if name1 in self._diffeq_names:
            self._diffeq_names[self._diffeq_names.index(name1)] = name2
        if name1 in self._diffeq_names_nonzero:
            self._diffeq_names_nonzero[self._diffeq_names_nonzero.index(name1)] = name2
        # Strings
        if name1 in self._string:
            self._string[name2] = self._string[name1]
            del self._string[name1]
        for name, value in self._string.iteritems():
            self._string[name] = re.sub("\\b" + name1 + "\\b", name2, value)
        # Namespaces
        if name1 in self._namespace:
            self._namespace[name2] = self._namespace[name1]
            del self._namespace[name1]

    def __getattr__(self, name):
        '''
        Returns the corresponding function.
        '''
        # bug with clustertools
        if name == 'as_array':
            raise Exception()
        return lambda ** kwd:self.apply(name, kwd)

    def __len__(self):
        '''
        Number of differential equations

        Note: is this still used?
        '''
        return len(self._diffeq_names)

    def __repr__(self):
        s = ''
        for var in self._diffeq_names:
            s += 'd' + var + '/dt = ' + self._string[var] + ' [diffeq]\n'
        for var in self._eq_names:
            if var in self._alias:
                typename = ' [alias]'
            else:
                typename = ' [eq]'
            s += var + ' = ' + self._string[var] + typename + '\n'
        return s

    def __reduce__(self):
        # To avoid recursion, we temporarily set the class to a trivial
        # class, this is restored at the end, and by _load_Equations_from_pickle
        # too
        self.__class__, cls = PickledEquations, self.__class__
        selfcopy = copy.copy(self)
        # We need to delete __builtins__ from all the namespaces as it is
        # not picklable, so we make copies
        selfcopy._namespace = copy.copy(selfcopy._namespace)
        for key in selfcopy._namespace.keys():
            selfcopy._namespace[key] = copy.copy(selfcopy._namespace[key])
            if '__builtins__' in selfcopy._namespace[key]:
                del selfcopy._namespace[key]['__builtins__']
        selfcopy._function = {}
        # Sometimes namespaces have numpy ufuncs in them, which are not
        # picklable, so we replace them with PickledUfunc objects which just
        # store their name, and _load_equations_from_pickle will extract them
        # from numpy again
        def replaceufunc(d):
            for k in d.keys():
                v = d[k]
                if isinstance(v, numpy.ufunc):
                    d[k] = PickledUfunc(v)
                if isinstance(v, dict):
                    replaceufunc(v)
        replaceufunc(selfcopy.__dict__)
        self.__class__ = cls
        return (_load_Equations_from_pickle, (selfcopy, cls))

class PickledUfunc(object):
    def __init__(self, ufunc):
        self.name = ufunc.__name__
    def get(self):
        return getattr(numpy, self.name)

class PickledEquations(object):
    pass

def _load_Equations_from_pickle(eqs, cls):
    def replaceufunc(d):
        for k in d.keys():
            v = d[k]
            if isinstance(v, PickledUfunc):
                d[k] = v.get()
            if isinstance(v, dict):
                replaceufunc(v)
    replaceufunc(eqs.__dict__)
    eqs.__class__ = cls
    eqs.prepare()
    return eqs

# Utilitary functions
# -------------------
def call_with_dict(f, d):
    '''
    Calls a function f with arguments from dictionary d.
    The dictionary can contain keys that are not variables of f.
    '''
    return f(*[d[var] for var in f.func_code.co_varnames])

def disjoint_dict_union(d1, d2):
    '''
    Merges the dictionaries d1 and d2 and checks that
    they are compatible (i.e., raises an exception if d1[key]!=d2[key])
    '''
    result = {}
    result.update(d1)
    for key, value in d2.iteritems(): # Bug here
        if (key in d1) and (d1[key] != value):
            raise AttributeError, "Incompatible dictionaries in disjoint union, problem with key " + key
        result[key] = value
    return result
