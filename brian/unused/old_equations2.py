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
from operator import isSequenceType
from units import DimensionMismatchError,second
from inspection import *
from string import Template
from optimiser import *
from scipy import exp
from scipy import weave
from globalprefs import *
import re
import inspect
from random import randint 

class Equations(object):
    """
    A container of equations, differential equations and aliases,
    with methods to manipulate them.
    Ex:
        eqs=Equations()
        eqs=Equations(Eq(u=lambda u,v:3*u*v+5,unit=mV),
                      DiffEq=(x=lambda x:-x,unit=nA),
                      Alias=(x='y'))
    """
    def __init__(self,*args,**kwds):
        if len(args) and isinstance(args[0],str):
            args = parse_string_equations(args[0],**kwds)
        # Empty object
        self._eq=[] # equations (couples (name,function))
        self._diffeq=[] # differential equations
        self._alias=[] # aliases
        self._units={} # dictionary of units
        self._useweave=get_global_preference('useweave')
        self._eqdict={} # dictionary of static equations
        self._diffeqdict={} # dictionary of dynamic equations
        for eq in args:
            if isinstance(eq,Equations):
                self.__iadd__(eq)
            else:
                raise TypeError,"Initialize with a (possibly empty) list of equations"
    
    def __add__(self,other):
        '''
        Union of two sets of equations
        '''
        result=Equations()
        result._eq=disjoint_eqlist_union(self._eq,other._eq)
        result._diffeq=disjoint_eqlist_union(self._diffeq,other._diffeq)
        result._alias=disjoint_eqlist_union(self._alias,other._alias)
        result._eqdict=disjoint_dict_union(self._eqdict,other._eqdict)
        result._diffeqdict=disjoint_dict_union(self._diffeqdict,other._diffeqdict)
        try:
            result._units=disjoint_dict_union(self._units,other._units)
        except AttributeError:
            raise DimensionMismatchError("The two sets of equations do not have compatible units")
        result.set_eq_order()
        return result
    
    def __iadd__(self,other):
        self._eq=disjoint_eqlist_union(self._eq,other._eq)
        self._diffeq=disjoint_eqlist_union(self._diffeq,other._diffeq)
        self._alias=disjoint_eqlist_union(self._alias,other._alias)
        self._eqdict=disjoint_dict_union(self._eqdict,other._eqdict)
        self._diffeqdict=disjoint_dict_union(self._diffeqdict,other._diffeqdict)
        try:
            self._units=disjoint_dict_union(self._units,other._units)
        except AttributeError:
            raise DimensionMismatchError("The two sets of equations do not have compatible units")
        self.set_eq_order()
        return self

    def optimize(self,n):
        '''
        Optimizes the equations using blitz.
        n = vector size (for arguments)
        '''
        for var,f in self._diffeq:
            self._diffeq[var][1]=optimize_lambda(f,n)
        for var,f in self._eq:
            self._eq[var][1]=optimize_lambda(f,n)

    def set_eq_order(self):
        '''
        Computes the internal depency graph of static variables
        and deduces the update order.
        Sets the list of dependencies of dynamic variables on static variables.
        This is called by check_units()
        ** TO CHANGE **
        '''
        if len(self._eq)>0:
            # Internal dependency dictionary
            dependency={}
            for key,value in self._eq:
                dependency[key]=[var for var in value.func_code.co_varnames if var in self._eqdict]
            
            # Sets the order
            staticvars_list=[]
            no_dep=None
            while (len(staticvars_list)<len(self._eq)) and (no_dep!=[]):
                no_dep=[key for key,value in dependency.iteritems() if value==[]]
                staticvars_list+=no_dep
                # Clear dependency list
                for key in no_dep:
                    del dependency[key]
                for key,value in dependency.iteritems():
                    dependency[key]=[var for var in value if not(var in staticvars_list)]
            
            if no_dep==[]: # The dependency graph has cycles!
                raise ReferenceError,"The static variables are referring to each other"
        else:
            staticvars_list=[]
            
        # Calculate dependencies on static variables
        self.staticdep={}
        for key in staticvars_list:
            self.staticdep[key]=[]
            for var in self._eqdict[key].func_code.co_varnames:
                if var in self._eqdict:
                    self.staticdep[key]+=[var]+self.staticdep[var]
        for key,f in self._diffeq:
            self.staticdep[key]=[]
            for var in f.func_code.co_varnames:
                if var in self._eqdict:
                    self.staticdep[key]+=[var]+self.staticdep[var]
        
        # Sort the dependency lists
        for key in self.staticdep:
            staticdep=[(staticvars_list.index(var),var) for var in self.staticdep[key]]
            staticdep.sort()
            self.staticdep[key]=[x[1] for x in staticdep]
            
        # Update _eq
        for i in range(len(staticvars_list)):
            var=staticvars_list[i]
            staticvars_list[i]=(var,self._eqdict[var])
        self._eq=staticvars_list

    def __contains__(self,var):
        # Change this? Add eq and alias?
        return (self._diffeq!=[] and (var in self._eqdict))
    
    def __len__(self):
        '''
        Number of differential equations
        '''
        return len(self._diffeq)
    
    def set_diffeq_order(self):
        '''
        Sorts the list of differential equations.
        Finds the variable that is most likely to be the
        membrane potential and sets it as the first equation.
        '''
        vm_names=['v','V','vm','Vm']
        guesses=[var for var,_ in self._diffeq if var in vm_names]
        if len(guesses)==1: # Unambiguous
            var_vm=guesses[0]
            # Set it as the first variable
            i=self._diffeq.index(var_vm)
            self._diffeq[0],self._diffeq[i]=self._diffeq[i],self._diffeq[0]
            self.found_vm=True
        else: # Ambiguous or not found
            self.found_vm=False

    def apply(self,state,vardict):
        '''
        Calculates self._diffeq[state] with arguments in vardict and
        static variables. The dictionary is filled with the required
        static variables.
        TODO: make it faster by overriding self._diffeq[var].
        '''
        f=self._diffeqdict[state]
        # Calculate static variables
        for var in self.staticdep[state]:
            vardict[var]=call_with_dict(self._eqdict[var],vardict)
        return f(*[vardict[var] for var in f.func_code.co_varnames])

    def variables(self):
        '''
        Returns the list of variables for all the equations, excluding
        static variables.
        '''
        result=[]
        for _,f in self._diffeq:
            result+=list(f.func_code.co_varnames)
            
        return [x for x in list(set(result)) if not(x in self._eqdict)]

    def is_conditionally_linear(self):
        '''
        Returns True if the equations are linear with respect to the
        state variable.
        '''
        for var,_ in self._diffeq:
            S=self._units.copy()
            S[var]=AffineFunction()
            #self.apply(var,S)
            try:
                self.apply(var,S)
            except:
                return False
        return True
    
    def forward_euler(self,S,dt):
        '''
        Updates the value of the state variables in dictionary S
        with the forward Euler algorithm over step dt.
        TODO: eliminate buffer for dependency graphs without cycles?
        '''
        # Calculate all static variables
        for var,f in self._eq:
            S[var]=call_with_dict(f,S)
        # Calculate derivatives
        buffer={}
        for varname,f in self._diffeq:
            buffer[varname]=f(*[S[var] for var in f.func_code.co_varnames])
        # Update variables
        for var,f in self._diffeq:
            S[var]+=dt*buffer[var]

    def exponential_euler(self,S,dt):
        '''
        Updates the value of the state variables in dictionary S
        with an exponential Euler algorithm over step dt.
        Test with is_conditionally_linear first.
        Same as default integration method in Genesis.
        Close to the implicit Euler method in Neuron.
        '''
        # Calculate all static variables (BAD: INSERT IT BELOW)
        for var,f in self._eq:
            S[var]=call_with_dict(f,S)
        n=len(S[self._diffeq[0][0]])
        # Calculate the coefficients of the affine function
        Z=zeros(n)
        O=ones(n)
        A={}
        B={}
        for varname,f in self._diffeq:
            oldval=S[varname]
            S[varname]=Z
            B[varname]=f(*[S[var] for var in f.func_code.co_varnames]).copy() # important if compiled
            S[varname]=O
            A[varname]=f(*[S[var] for var in f.func_code.co_varnames])-B[varname]
            B[varname]/=A[varname]
            S[varname]=oldval
        # Integrate
        for varname,f in self._diffeq:
            if self._useweave:
                Bx=B[varname]
                Ax=A[varname]
                Sx=S[varname]
                # Compilation with blitz: we need an approximation because exp is not understood
                #weave.blitz('Sx[:]=-Bx+(Sx+Bx)*(1.+Ax*dt*(1.+.5*Ax*dt))',check_size=0)
                code =  """
                for(int k=0;k<n;k++)
                    Sx(k)=-Bx(k)+(Sx(k)+Bx(k))*exp(Ax(k)*dt);
                """
                weave.inline(code,['n','Bx','Sx','Ax','dt'],\
                             compiler='gcc',
                             type_converters=weave.converters.blitz)
            else:
                #S[varname][:]=-B[varname]+(S[varname]+B[varname])*exp(A[varname]*dt)
                # A little faster:
                S[varname]+=B[varname]
                S[varname]*=exp(A[varname]*dt)
                S[varname]-=B[varname]
    
    def check_units(self):
        '''
        Checks the units of the differential equations, using
        the units of x.
        df_i/dt must have units of x_i / time.
        '''
        self.set_eq_order()
        # Units of static variables
        for var,f in self._eq:
            self._units[var]=call_with_dict(f,self._units)
        try:
            for var,f in self._diffeq:
                f.func_globals['xi']=0*second**-.5 # Noise
                self.apply(var,self._units)+(self._units[var]/second) # Check that the two terms have the same dimension
        except DimensionMismatchError,inst:
            raise DimensionMismatchError("The differential equations are not homogeneous!",*inst._dims)

class Equation(Equations): # or Equations?
    """
    An equation object.
    Example:
        eq=Equation(i=lambda vr,v:(v-vr)/R,unit=mV)
    Units are mandatory.
    """
    def __init__(self,**kwargs):
        Equations.__init__(self)
        if 'unit' not in kwargs:
            raise AttributeError,"The units must be specified"
        if len(kwargs)>2:
            raise AttributeError,"Too many arguments"
        # Find variable name
        arg1,arg2=kwargs
        if arg1=='unit':
            name=arg2
        else:
            name=arg1
        self._eq.append((name,kwargs[name]))
        self._units[name]=kwargs['unit']
        self._eqdict[name]=kwargs[name]

class DifferentialEquation(Equations):
    """
    A differential equation.
    Example:
        eq=DifferentialEquation(dv=lambda v:-v/tau,unit=mV)
    Units are mandatory.
    """
    def __init__(self,**kwargs):
        Equations.__init__(self)
        if 'unit' not in kwargs:
            raise AttributeError,"The units must be specified"
        if len(kwargs)>2:
            raise AttributeError,"Too many arguments"
        # Find variable name
        arg1,arg2=kwargs
        if arg1=='unit':
            name=arg2
        else:
            name=arg1
        # Check the syntax (d.)
        if name=='d' or name[0]!='d':
            raise AttributeError,"Syntax: DifferentialEquation(dv=...,unit=...)"
        name=name[1:]
        self._diffeq.append((name,kwargs['d'+name]))
        self._units[name]=kwargs['unit']
        self._diffeqdict[name]=kwargs['d'+name]

class Alias(Equations):
    """
    An alias for a variable.
    Ex:
        eq=Alias(u='v')
    """
    def __init__(self,**kwargs):
        Equations.__init__(self)
        if len(kwargs)!=1:
            raise AttributeError,"Syntax: Alias(u='v')"
        name,=kwargs
        self._alias.append((name,kwargs[name]))
        self._units[name]=None # or do not set it?
        self._eq.append((name,eval(Template('lambda $x:$x').substitute(x=kwargs[name]))))
        self._eqdict[name]=self._eq[0][1]

def call_with_dict(f,d):
    '''
    Calls a function f with arguments from dictionary d.
    The dictionary can contain keys that are not variables of f.
    '''
    return f(*[d[var] for var in f.func_code.co_varnames])

def disjoint_dict_union(d1,d2):
    '''
    Merges the dictionaries d1 and d2 and checks that
    they are compatible (i.e., raises an exception if d1[key]!=d2[key])
    '''
    result={}
    result.update(d1)
    for key,value in d2.iteritems(): # Bug here
        if (key in d1) and (d1[key]!=value):
            raise AttributeError,"Incompatible dictionaries in disjoint union."
        result[key]=value
    return result

def disjoint_eqlist_union(d1,d2):
    '''
    Merges the equation lists d1 and d2 and checks that
    they are compatible (lists of couples (name,value)).
    '''
    result=[]+d1 # copies the list
    names=[name for name,_ in d1]
    for key,value in d2: # Bug here
        if (key in names):
            if (d1[names.index(key)][1]!=value):
                raise AttributeError,"Incompatible dictionaries in disjoint union."
        else:
            result.append((key,value))
    return result

def parse_string_equations(eqns,level=2,verbose=True,namespace=None,extra_arg_vars=None):
    ns = {}
    if namespace is None:
        ns.update(inspect.stack()[level][0].f_globals)
        ns.update(inspect.stack()[level][0].f_locals)
    else:
        ns.update(**namespace)
    print eqns
    parsed = []
    preparsed = []
    basename = '_basename_parse_string_equations' + str(randint(10000,100000))
    for line in eqns.split('\n'):
        line = line.strip()
        if len(line):
            m = re.search('d(.*?)\s*/\s*dt\s*=\s*(.*?)\s*:\s*(.*)',line)
            if m is not None:
                preparsed.append(('de',m.group(1),m.group(2),m.group(3)))
            else:
                m = re.search('([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*$',line)
                if m is not None:
                    preparsed.append(('alias',m.group(1),m.group(2),''))
                else:
                    m = re.search('([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*(.*?)\s*:\s*(.*)',line)
                    if m is not None:
                        preparsed.append(('eqn',m.group(1),m.group(2),m.group(3)))
                    else:
                        m = re.search('param(?:eter)?s?(.*)',line)
                        if m is not None:
                            s = m.group(1)
                            l = s.split(',')
                            l = [_.strip() for _ in l]
                            for s in l:
                                if ':' in s:
                                    m = re.search('\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*:\s*(.*)',s)
                                    if m is not None:
                                        preparsed.append(('param',m.group(1),'',m.group(2)))
                                else:
                                    preparsed.append(('param',s,'',''))
    #dynvars = [varname for eqtype, varname, valstr, unit in preparsed if eqtype=='de' or eqtype=='param']
    argvars = [varname for eqtype, varname, valstr, unit in preparsed]
    if extra_arg_vars is not None:
        argvars.extend(extra_arg_vars)
    ns.update({basename+'DifferentialEquation':DifferentialEquation,
               basename+'Equation':Equation,
               basename+'Alias':Alias})
    for eqtype, varname, valstr, unit in preparsed:
        if eqtype=='de':
            argvarstr = ','.join([argvarname for argvarname in argvars if argvarname in valstr])
            s = basename+'de = '+basename+'DifferentialEquation(d' + varname + '=lambda ' + argvarstr + ': ' + valstr+',unit='+unit+')'
            if verbose:
                print s.replace(basename,'').replace('de = ','')
            exec s in ns 
            parsed.append(ns[basename+'de'])
        elif eqtype=='eqn':
            argvarstr = ','.join([argvarname for argvarname in argvars if argvarname in valstr])
            s = basename+'eqn = '+basename+'Equation(' + varname + '=lambda ' + argvarstr + ': ' + valstr+',unit='+unit+')'
            if verbose:
                print s.replace(basename,'').replace('eqn = ','')
            exec s in ns 
            parsed.append(ns[basename+'eqn'])
        elif eqtype=='alias':
            s = basename+'alias = '+basename+'Alias(' + varname + "='" + valstr + "')"
            if verbose:
                print s.replace(basename,'').replace('alias = ','')
            exec s in ns
            parsed.append(ns[basename+'alias'])
    return parsed

# Shorter names
#Eq=Equation
#DiffEq=DifferentialEquation
