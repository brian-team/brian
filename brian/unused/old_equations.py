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

class Equations(object):
        '''
        Example:
        eqs=Equations(dv=lambda v,g:g*(E-v),dg=lambda g:-g/tau)
        eqs['v'] returns lambda v,g:g*(E-v)
        eqs['g2']=lambda g2:g2*(E-v) inserts another equation
        
        Units:
        eqs=Equations(dv=lambda v,g:g*(E-v),dg=lambda g:-g/tau,units={'v':volt,'g':siemens})
        or eqs.units={'v':volt,'g':siemens}
        or eqs.units['v']=volt; eqs.units['g']=siemens
        
        eqs.dynamicvars is the list of dynamic variable names, starting with the
        membrane potential (if found and unambiguous, as stated by eqs.found_vm).
        '''
        def __init__(self,*arglist,**args):
            if (len(arglist)>0) and (isSequenceType(arglist[0])): # A tuple is passed
                model=arglist[0]
                vars=getvarnames(model)
                # Check the number of equations
                if len(model)>len(vars):
                    raise TypeError,"More equations than variables!"
                for var,eq in zip(vars,model):
                    self._eq[var]=eq
                                        
            # Get names of dynamic variables
            dynamic_vars=[key[1:] for key in args.keys() if key[0]=='d']
            self._eq={} # dictionary of equations
            for var in dynamic_vars:
                self[var]=args['d'+var]
                
            # Units
            if 'units' in args:
                self.units=args['units']
            else:
                self.units={}
                
            # Ordered list of dynamical variables
            self.set_dynamic_vars()
            
            # Static variables
            self.staticvars={}
            self.staticvars_list=[] # Update order
            
            # Aliases
            self._aliases={}
            
            self._useweave=get_global_preference('useweave')
        
        def optimize(self,n):
            '''
            Optimizes the equations using blitz.
            n = vector size (for arguments)
            '''
            for var,f in self._eq.iteritems():
                self._eq[var]=optimize_lambda(f,n)
            for var,f in self.staticvars.iteritems():
                self.staticvars[var]=optimize_lambda(f,n)
        
        def set_alias(self,var1,var2):
            '''
            Sets var1 as an alias for variable var2.
            '''
            if var1!=var2:
                self._aliases[var1]=var2
                self.staticvars[var1]=eval(Template('lambda $x:$x').substitute(x=var2))
        
        def set_static_vars_order(self):
            '''
            Computes the internal depency graph of static variables
            and deduces the update order.
            Sets the list of dependencies of dynamic variables on static variables.
            This is called by check_units()
            '''
            if len(self.staticvars)>0:
                # Internal dependency dictionary
                dependency={}
                for key,value in self.staticvars.iteritems():
                    dependency[key]=[var for var in value.func_code.co_varnames if var in self.staticvars]
                
                # Sets the order
                self.staticvars_list=[]
                no_dep=None
                while (len(self.staticvars_list)<len(self.staticvars)) and (no_dep!=[]):
                    no_dep=[key for key,value in dependency.iteritems() if value==[]]
                    self.staticvars_list+=no_dep
                    # Clear dependency list
                    for key in no_dep:
                        del dependency[key]
                    for key,value in dependency.iteritems():
                        dependency[key]=[var for var in value if not(var in self.staticvars_list)]
                
                if no_dep==[]: # The dependency graph has cycles!
                    raise ReferenceError,"The static variables are referring to each other"
            else:
                self.staticvars_list=[]
                
            # Calculate dependencies on static variables
            self.staticdep={}
            for key in self.staticvars_list:
                self.staticdep[key]=[]
                for var in self.staticvars[key].func_code.co_varnames:
                    if var in self.staticvars:
                        self.staticdep[key]+=[var]+self.staticdep[var]
            for key,f in self._eq.iteritems():
                self.staticdep[key]=[]
                for var in f.func_code.co_varnames:
                    if var in self.staticvars:
                        self.staticdep[key]+=[var]+self.staticdep[var]
                
            # Sort the dependency lists
            for key in self.staticdep:
                staticdep=[(self.staticvars_list.index(var),var) for var in self.staticdep[key]]
                staticdep.sort()
                self.staticdep[key]=[x[1] for x in staticdep]
                    
        def __getitem__(self,var):
            return self._eq[var]
        
        def __setitem__(self,var,f):
            self._eq[var]=f
            self.set_dynamic_vars()
            
        def __add__(self,other):
            '''
            Union of two sets of equations
            '''
            result=Equations()
            result._eq=disjoint_dict_union(self._eq,other._eq)
            try:
                result.units=disjoint_dict_union(self.units,other.units)
            except AttributeError:
                raise DimensionMismatchError("The two sets of equations do not have compatible units")
            result.staticvars=disjoint_dict_union(self.staticvars,other.staticvars)
            result.set_dynamic_vars()
            result._aliases=disjoint_dict_union(self._aliases,other._aliases)
            return result
        
        def __contains__(self,var):
            return (var in self._eq)
        
        def __len__(self):
            '''
            Number of equations
            '''
            return len(self._eq)
        
        def set_dynamic_vars(self):
            '''
            Sets a list of dynamic variables.
            Finds the variable that is most likely to be the
            membrane potential and sets it as the first equation.
            '''
            self.dynamicvars=self._eq.keys()
            vm_names=['v','V','vm','Vm']
            guesses=[var for var in self._eq.iterkeys() if var in vm_names]
            if len(guesses)==1: # Unambiguous
                var_vm=guesses[0]
                # Set it as the first variable
                i=self.dynamicvars.index(var_vm)
                self.dynamicvars[0],self.dynamicvars[i]=self.dynamicvars[i],self.dynamicvars[0]
                self.found_vm=True
            else: # Ambiguous or not found
                self.found_vm=False
        
        def apply(self,state,vardict):
            '''
            Calculates self._eq[state] with arguments in vardict and
            static variables. The dictionary is filled with the required
            static variables.
            TODO: make it faster by overriding self._eq[var].
            '''
            f=self._eq[state]
            # Calculate static variables
            for var in self.staticdep[state]:
                vardict[var]=call_with_dict(self.staticvars[var],vardict)
            return f(*[vardict[var] for var in f.func_code.co_varnames])
        
        def variables(self):
            '''
            Returns the list of variables for all the equations, excluding
            static variables.
            '''
            result=[]
            for f in self._eq.itervalues():
                result+=list(f.func_code.co_varnames)
                
            return [x for x in list(set(result)) if not(x in self.staticvars)]
        
        def is_conditionally_linear(self):
            '''
            Returns True if the equations are linear with respect to the
            state variable.
            '''
            for var in self.dynamicvars:
                S=self.units.copy()
                S[var]=AffineFunction()
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
            for var in self.staticvars_list:
                S[var]=call_with_dict(self.staticvars[var],S)
            # Calculate derivatives
            buffer={}
            for varname,f in self._eq.iteritems():
                buffer[varname]=f(*[S[var] for var in f.func_code.co_varnames])
            # Update variables
            for var,f in self._eq.iteritems():
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
            for var in self.staticvars_list:
                S[var]=call_with_dict(self.staticvars[var],S)
            n=len(S[self.dynamicvars[0]])
            # Calculate the coefficients of the affine function
            Z=zeros(n)
            O=ones(n)
            A={}
            B={}
            for varname,f in self._eq.iteritems():
                oldval=S[varname]
                S[varname]=Z
                B[varname]=f(*[S[var] for var in f.func_code.co_varnames]).copy() # important if compiled
                S[varname]=O
                A[varname]=f(*[S[var] for var in f.func_code.co_varnames])-B[varname]
                B[varname]/=A[varname]
                S[varname]=oldval
            # Integrate
            for varname,f in self._eq.iteritems():
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
            self.set_static_vars_order()
            # Units of static variables
            for var in self.staticvars_list:
                self.units[var]=call_with_dict(self.staticvars[var],self.units)
            try:
                for var,f in self._eq.iteritems():
                    f.func_globals['xi']=0*second**-.5 # Noise
                    self.apply(var,self.units)+(self.units[var]/second) # Check that the two terms have the same dimension
            except DimensionMismatchError,inst:
                raise DimensionMismatchError("The differential equations are not homogeneous!",*inst._dims)
        
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
