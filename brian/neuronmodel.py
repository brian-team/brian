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
"""
Neuron model base class : DEPRECATED
"""

__all__ = ['Model']

from units import *
import neurongroup
import magic
from equations import *
from brian_unit_prefs import bup
import warnings

class Model(object):
    """
    Stores properties that define a model neuron
    
    **NOTE: this class has been deprecated as of Brian 1.1**
    
    The purpose of this class is to store the parameters that define
    a model neuron, but not actually create any neurons themselves. That is
    done by the :class:`NeuronGroup` object. The parameters for initialising
    this object are the same as for :class:`NeuronGroup` less ``N`` and
    with the addition of ``equation`` and ``equations`` as
    alternative keywords for ``model`` (for readability of code).
    
    At the moment, this object simply stores a copy of these keyword
    assignments and passes them on to a :class:`NeuronGroup` when you
    instantiate it with this model, so the definitive reference point is
    the :class:`NeuronGroup`. For convenience, we include a copy of these
    arguments to initiate a :class:`Model` here:
    
    ``model``, ``equation`` or ``equations``
        An object defining the neuron model. It can be
        an :class:`Equations` object, a string defining an :class:`Equations` object,
        a :class:`StateUpdater` object, or a list or tuple of :class:`Equations` and
        strings.
    ``threshold=None``
        A :class:`Threshold` object, a function or a scalar quantity.
        If ``threshold`` is a function with one argument, it will be
        converted to a :class:`SimpleFunThreshold`, otherwise it will be a
        :class:`FunThreshold`. If ``threshold`` is a scalar, then a constant
        single valued threshold with that value will be used. In this case,
        the variable to apply the threshold to will be guessed. If there is
        only one variable, or if you have a variable named one of
        ``V``, ``Vm``, ``v`` or ``vm`` it will be used.
    ``reset=None``
        A :class:`Reset` object, a function or a scalar quantity. If it's a
        function, it will be converted to a :class:`FunReset` object. If it's
        a scalar, then a constant single valued reset with that value will
        be used. In this case,
        the variable to apply the reset to will be guessed. If there is
        only one variable, or if you have a variable named one of
        ``V``, ``Vm``, ``v`` or ``vm`` it will be used.
    ``refractory=0*ms``
        A refractory period, used in combination with the ``reset`` value
        if it is a scalar.
    ``order=1``
        The order to use for nonlinear differential equation solvers.
        TODO: more details.
    ``implicit=False``
        Whether to use an implicit method for solving the differential
        equations. TODO: more details.
    ``max_delay=0*ms``
        The maximum allowable delay (larger values use more memory).
        TODO: more details.
    ``compile=False``
        Whether or not to attempt to compile the differential equation
        solvers into ``C++`` code.
    ``freeze=False``
        If True, parameters are replaced by their values at the time
        of initialization.

    **Usage**
    
    You can either pass a :class:`Model` as an argument to initialise a
    :class:`NeuronGroup` or initialise a :class:`NeuronGroup` by writing::
    
        group = model * N
    
    to create a :class:`NeuronGroup` of ``N`` neurons based on that model.

    **Example**

    Starting with a model defined like this::
    
        model = Model(equations='''
        dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
        dge/dt = -ge/(5*ms) : volt
        dgi/dt = -gi/(10*ms) : volt
        ''', threshold=-50*mV, reset=-60*mV)
    
    The following two lines are equivalent::
    
        P = NeuronGroup(4000, model=model)
        P = 4000*model
    """
    @check_units(refractory=second,maxdelay=second)
    def __init__(self,**kwds):
        warnings.warn('Model object deprecated as of Brian 1.1', DeprecationWarning)
        # todo: define a set of acceptable keywords and delete the rest
        if 'equation' in kwds: kwds['model']=kwds.pop('equation')
        if 'equations' in kwds: kwds['model']=kwds.pop('equations')
        self.kwds = kwds
        if isinstance(kwds['model'],str):
            # level=3 looks pretty strange, but the point is to grab the appropriate namespace from the
            # frame where the string is defined, rather than from the current frame which is what
            # Equations will do by default. The level has to be 3 because it is 1+2, where the 1 is
            # the frame that called the Model init, and the 2 is the 2 decorators added to the
            # beginning of the __init__ method.
            if bup.use_units:
                kwds['model']=Equations(kwds['model'],level=2)
            else:
                kwds['model']=Equations(kwds['model'],level=1)
        G = neurongroup.NeuronGroup(N=1,model=self) # just to make sure it can be done
    
    @magic.magic_return
    def __mul__(self, N):
        model = self.kwds['model']
        if isinstance(model,(str,list,tuple)):
            model = Equations(model, level=2)
        self.kwds['model'] = model       
        return neurongroup.NeuronGroup(N, model=self)
    __rmul__ = __mul__
