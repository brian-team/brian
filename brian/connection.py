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
__all__=[
         'Connection', 'IdentityConnection', 'MultiConnection',
         #'HeterogeneousDelayConnection', # this class is not defined
         'random_matrix_fixed_column',
         'random_matrix', 'UserComputedConnectionMatrix',
         'UserComputedSparseConnectionMatrix', 'random_row_func',
         'random_sparse_row_func'
         ]

import copy
from itertools import izip
from random import sample
import bisect
from units import *
import types
import magic
from log import *
from numpy import *
from scipy import sparse,stats,rand,weave,linalg
import numpy
import random as pyrandom
from scipy import random as scirandom
from utils.approximatecomparisons import is_within_absolute_tolerance
from globalprefs import *
    
effective_zero = 1e-40

#TODO: connect -> setitem

class ConnectionMatrix(object):
    """
    Connection matrix: a specific type of matrix
    for synaptic connections.
    """
    def add_row(self,i,X):
        X+=self[i] # row should be a view on a vector
    
    def add_rows(self,spikes,X):
        for i in spikes:
            self.add_row(i,X)

    def add_scaled_row(self,i,X,factor):
        X+=factor*self[i]
    
    def set_row(self,i,X):
        self[i]=X
        
    def freeze(self):
        """
        Converts the matrix to a faster structure.
        """
        pass

class DenseConnectionMatrix(ConnectionMatrix,ndarray):
    """
    A dense connection matrix.
    This is the default matrix for plastic synapses.
    """
    def __init__(self, dims, **kwds):
        numpy.ndarray.__init__(self, dims, **kwds)
        self[:]=0
        self._useaccel = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
        
    def __setitem__(self, index, W):
        # Make it work for sparse matrices
        if isinstance(W,sparse.spmatrix):
            ndarray.__setitem__(self,index,W.todense())
        else:
            ndarray.__setitem__(self,index,W)
    
    def add_row(self,i,X):
        X+=self[i,:] # row should be a view on a vector
    
    def add_rows(self,spikes,X):
        if self._useaccel:
            # TODO: do a C++ version for add_scaled_rows.
            if not isinstance(spikes, numpy.ndarray):
                spikes = array(spikes, dtype=int)
            nspikes = len(spikes)
            N = len(X)
            selfarray = asarray(self)
            code =  """
                    for(int i=0;i<nspikes;i++)
                    {
                        int k = spikes(i);
                        for(int j=0;j<N;j++)
                            X(j)+=selfarray(k,j);
                    }
                    """
            weave.inline(code,['selfarray','X','spikes','nspikes','N'],
                         compiler=self._cpp_compiler,
                         type_converters=weave.converters.blitz,
                         extra_compile_args=['-O3'])
        else:
            for i in spikes:
                X+=self[i,:] # row should be a view on a vector

    def add_scaled_row(self,i,X,factor):
        X+=factor*self[i,:]
        
    def freeze(self):
        """
        Converts the matrix to a faster structure.
        """
        pass

def is_colon_slice(item):
    return isinstance(item,slice) and item.start==None and item.step==None and item.stop==None

# TODO: use own structure?
class SparseConnectionMatrix(ConnectionMatrix,sparse.lil_matrix):
    """
    A sparse connection matrix, i.e., zero entries are not stored.
    This is the default matrix for static synapses.
    """
    def __init__(self, dims, **kwds):
        sparse.lil_matrix.__init__(self, dims, **kwds)
        self._useaccel = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
        self.rowret = numpy.zeros(dims[1])

    # This stuff actually speeds things up considerably, but it needs
    # standardising...
#    def get_row(self, i):
#        self.rowret[:]=0
#        self.rowret[self.rows[i]] = self.data[i]
#        return self.rowret
#    def __getitem__(self, item):
#        if isinstance(item,slice):
#            if is_colon_slice(item):
#                return self.alldata
#            else:
#                raise ValueError(str(item)+' not supported.')
#        if isinstance(item,int):
#            return self.get_row(item)
#        if isinstance(item,tuple):
#            if len(item)!=2:
#                raise TypeError('Only 2D indexing supported.')
#            item_i, item_j = item
#            if isinstance(item_i, int) and isinstance(item_j, slice):
#                if is_colon_slice(item_j):
#                    return self.get_row(item_i)
#                raise ValueError('Only ":" indexing supported.')
#            if isinstance(item_i, slice) and isinstance(item_j, int):
#                if is_colon_slice(item_i):
#                    return self.get_col(item_j)
#                raise ValueError('Only ":" indexing supported.')
#            if isinstance(item_i, int) and isinstance(item_j, int):
#                pointer = self.get_pointer(item_i, item_j)
#                if pointer is None:
#                    return 0.0
#                return self.alldata[pointer]
#            raise TypeError('Only (i,:), (:,j) and (i,j) indexing supported.')
#        raise TypeError('Can only get items of type slice or tuple')

        
    def __setitem__(self, index, W):
        """
        Speed-up if x is a sparse matrix.
        TODO: checks (first remove the data).
        """
        try:
            i, j = index
        except (ValueError, TypeError):
            raise IndexError, "invalid index"

        if isinstance(i, slice) and isinstance(j,slice) and\
           (i.step is None) and (j.step is None) and\
           (isinstance(W,sparse.lil_matrix) or isinstance(W,ndarray)):
            rows = self.rows[i]
            datas = self.data[i]
            j0=j.start
            if isinstance(W,sparse.lil_matrix):
                for row,data,rowW,dataW in izip(rows,datas,W.rows,W.data):
                    jj=bisect.bisect(row,j0) # Find the insertion point
                    row[jj:jj]=[j0+k for k in rowW]
                    data[jj:jj]=dataW
            elif isinstance(W,ndarray):
                nq=W.shape[1]
                for row,data,rowW in izip(rows,datas,W):
                    jj=bisect.bisect(row,j0) # Find the insertion point
                    row[jj:jj]=range(j0,j0+nq)
                    data[jj:jj]=rowW
        else:
            sparse.lil_matrix.__setitem__(self,index,W)

    def add_row(self,i,X):
        X[self.rows[i]]+=self.data[i]

    def add_rows(self,spikes,X):
        if self._useaccel:
            # TODO: redesign data types to make this more efficient
            # TODO: do a C++ version for add_scaled_rows.
            if not isinstance(spikes, numpy.ndarray):
                spikes = array(spikes, dtype=int)
            if isinstance(self.rows,numpy.ndarray):
                self.rows = self.rows.tolist()
                self.data = self.data.tolist()
            if not isinstance(self.rows[0],numpy.ndarray):
                self.freeze()
            rows = self.rows
            datas = self.data # bad English but makes below clearer
            nspikes = len(spikes)
            # Brief explanation of the code below:
            # rows and datas are Python lists of numpy arrays, in the case of rows
            # the numpy arrays have dtype=int, and in the case of datas dtype=float.
            # The notation rows[j] in the C++ code below returns a PyObject* object
            # from the rows variable which is a py::list object. The convert_to_numpy
            # function converts the PyObject* object to a PyArrayObject* (which is a
            # numpy standard data type). Note that if the connection hasn't been frozen
            # this code will fail, and it will appear to Python as if weave.inline
            # failed. Then follows two checks. Note that all of this comes from the
            # generated code if you just use the weave.blitz converters on a numpy
            # array. We have to do it ourselves here because we have a list of arrays
            # which aren't named, so we can't use the blitz converters for it.
            # Finally, we convert to a blitz::Array object so that we can use the
            # notation row(k) to refer to the kth element of the row in the for
            # loop at the end. This could be made considerably more efficient by
            # having a single static array containing all the rows and data, with
            # pointers to the starts of each row. This would mean it couldn't be
            # modified after it was created though (although this is true now if
            # you use the freeze() function).
            code =  """
                    for(int i=0;i<nspikes;i++)
                    {
                        int j = spikes(i);
                        PyArrayObject* _row = convert_to_numpy(rows[j], "row");
                        conversion_numpy_check_type(_row, PyArray_INT, "row");
                        conversion_numpy_check_size(_row, 1, "row");
                        blitz::Array<int,1> row = convert_to_blitz<int,1>(_row,"row");
                        PyArrayObject* _data = convert_to_numpy(datas[j], "data");
                        conversion_numpy_check_type(_data, PyArray_DOUBLE, "data");
                        conversion_numpy_check_size(_data, 1, "data");
                        blitz::Array<double,1> data = convert_to_blitz<double,1>(_data,"data");
                        int m = row.numElements();
                        for(int k=0;k<m;k++)
                        {
                            X(row(k)) += data(k);
                        }
                    }
                    """
            weave.inline(code,['X','rows','datas','spikes','nspikes'],
                         compiler=self._cpp_compiler,
                         type_converters=weave.converters.blitz,
                         extra_compile_args=['-O3'])
        else:
            for i in spikes:
                X[self.rows[i]]+=self.data[i]

    def add_scaled_row(self,i,X,factor):
        X[self.rows[i]]+=factor*self.data[i]

    def freeze(self):
        '''
        Converts the connection matrix to a faster structure.
        Replaces array of lists (= lil_matrix) by array of arrays.
        N.B.: that's a hack (many methods will probably not work anymore).
        '''
        for i in range(self.shape[0]):
            self.rows[i]=array(self.rows[i],dtype=int)
            self.data[i]=array(self.data[i])

class ComputedConnectionMatrix(ConnectionMatrix):
    """
    A connection matrix that is computed, i.e., no storing.
    Synaptic plasticity is not possible with these matrices.
    """
    pass

def random_row_func(N, p, weight=1., initseed=None):
    '''
    Returns a random connectivity ``row_func`` for use with :class:`UserComputedConnectionMatrix`
    
    Gives equivalent output to the :meth:`Connection.connect_random` method.
    
    Arguments:
    
    ``N``
        The number of target neurons.
    ``p``
        The probability of a synapse.
    ``weight``
        The connection weight (must be a single value).
    ``initseed``
        The initial seed value (for reproducible results).
    '''
    if initseed is None:
        initseed = pyrandom.randint(100000,1000000) # replace this
    cur_row = numpy.zeros(N)
    myrange = numpy.arange(N, dtype=int)
    def row_func(i):
        pyrandom.seed(initseed+int(i))
        scirandom.seed(initseed+int(i))
        k = scirandom.binomial(N, p, 1)[0]
        cur_row[:] = 0.0
        cur_row[pyrandom.sample(myrange,k)] = weight
        return cur_row
    return row_func

class UserComputedConnectionMatrix(ConnectionMatrix):
    '''
    A computed connection matrix defined by a user-specified function
    
    Normally this matrix will be initialised by passing the class
    object to the :class:`Connection` object. In the initialisation
    of the :class:`Connection` specify ``structure=UserComputedConnectionMatrix``
    and add the keyword ``row_func=...``, e.g.::
    
        def f(i):
            return max_weight*ones(N)/(1+(arange(N)-i)**2)
        C = Connection(G1, G2, structure=UserComputedConnectionMatrix, row_func=f)
    
    Initialisation arguments:
    
    ``dims``
        The pair ``(N,M)`` specifying the dimensions of the matrix.
    ``row_func``
        The function ``f(i)`` which returns an array of length ``M``,
        the weight matrix for row ``i``. Note that you are responsible
        for making sure the function returns consistent results (so
        random functions should be initialised with a seed based on
        the row ``i``).
    
    **Limitations**
    
    This type of connection matrix cannot be changed during a run, and
    cannot be used with methods like :class:`Connection.connect_random`.
    
    **Efficiency considerations**
    
    This connection matrix is for dense connectivity, if the connectivity
    is sparse you might get better performance with :class:`UserComputedSparseConnectionMatrix`.
    '''
    def __init__(self, dims, row_func):
        self.sourcelen, self.targetlen = dims
        self.row_func = row_func
    def get_row(self, i):
        return self.row_func(i)
    def __getitem__(self, item):
        if isinstance(item,int):
            return self.get_row(item)
        if isinstance(item,tuple):
            if len(item)==2:
                item_i, item_j = item
                if isinstance(item_i, int) and isinstance(item_j, slice):
                    if is_colon_slice(item_j):
                        return self.get_row(item_i)
        raise ValueError('Only "i,:" indexing supported.')

def random_sparse_row_func(N, p, weight=1., initseed=None):
    '''
    Returns a random connectivity ``row_func`` for use with :class:`UserComputedSparseConnectionMatrix`
    
    Gives equivalent output to the :meth:`Connection.connect_random` method.
    
    Arguments:
    
    ``N``
        The number of target neurons.
    ``p``
        The probability of a synapse.
    ``weight``
        The connection weight (must be a single value).
    ``initseed``
        The initial seed value (for reproducible results).
    '''
    if initseed is None:
        initseed = pyrandom.randint(100000,1000000) # replace this
    myrange = numpy.arange(N, dtype=int)
    def row_func(i):
        pyrandom.seed(initseed+int(i))
        scirandom.seed(initseed+int(i))
        k = scirandom.binomial(N, p, 1)[0]
        return (pyrandom.sample(myrange,k), weight)
    return row_func

class UserComputedSparseConnectionMatrix(ConnectionMatrix):
    '''
    A computed sparse connection matrix defined by a user-specified function
    
    Normally this matrix will be initialised by passing the class
    object to the :class:`Connection` object. In the initialisation
    of the :class:`Connection` specify ``structure=UserComputedSparseConnectionMatrix``
    and add the keyword ``row_func=...``, e.g.::
    
        def f(i):
            if 0<i<N-1:
                return ([i-1,i+1], weight*ones(2))
            elif i>0:
                return ([i-1], weight*ones(1))
            else:
                return ([i+1], weight*ones(1))
        C = Connection(G1, G2, structure=UserComputedSparseConnectionMatrix, row_func=f)
    
    Initialisation arguments:
    
    ``dims``
        The pair ``(N,M)`` specifying the dimensions of the matrix.
    ``row_func``
        The function ``f(i)`` which for a row ``i`` returns a pair ``(indices, values))``
        consisting of a list or array ``indices`` with the indices of the
        nonzero elements of the row, and an array of the same length ``values``
        giving the weight matrix for those indices. Note that you are responsible
        for making sure the function returns consistent results (so
        random functions should be initialised with a seed based on
        the row ``i``).
    
    **Limitations**
    
    This type of connection matrix cannot be changed during a run, and
    cannot be used with methods like :class:`Connection.connect_random`.
    
    **Efficiency considerations**
    
    This connection matrix is for sparse connectivity, if the connectivity
    is dense you might get better performance with :class:`UserComputedConnectionMatrix`.
    '''
    def __init__(self, dims, row_func):
        self.sourcelen, self.targetlen = dims
        self.row_func = row_func
        self.cur_row = numpy.zeros(dims[1])
    def add_row(self,i,X):
        indices, values = self.row_func(i)
        X[indices]+=values
    def add_scaled_row(self,i,X,factor):
        # modulation may not work? need factor[self.rows[i]] here? is factor a number or an array?
        X[indices]+=factor*values
    def get_row(self, i):
        indices, values = self.row_func(i)
        self.cur_row[:] = 0.0
        self.cur_row[indices] = values
        return self.cur_row
    def __getitem__(self, item):
        if isinstance(item,int):
            return self.get_row(item)
        if isinstance(item,tuple):
            if len(item)==2:
                item_i, item_j = item
                if isinstance(item_i, int) and isinstance(item_j, slice):
                    if is_colon_slice(item_j):
                        return self.get_row(item_i)
        raise ValueError('Only "i,:" indexing supported.')

    
#TODO: unit checking for some functions
class Connection(magic.InstanceTracker):
    '''
    Mechanism for propagating spikes from one group to another

    A Connection object declares that when spikes in a source
    group are generated, certain neurons in the target group
    should have a value added to specific states. See
    Tutorial 2: Connections to understand this better.

    **Initialised as:** ::
    
        Connection(source, target[, state=0[, delay=0*ms[, modulation=None]]])
    
    With arguments:
    
    ``source``
        The group from which spikes will be propagated.
    ``target``
        The group to which spikes will be propagated.
    ``state``
        The state variable name or number that spikes will be
        propagated to in the target group.
    ``delay``
        The delay between a spike being generated at the source
        and received at the target. At the moment, the mechanism
        for delays only works for relatively short delays (an
        error will be generated for delays that are too long).
    ``modulation``
        The state variable name from the source group that scales
        the synaptic weights (for short-term synaptic plasticity).
    ``structure``
        Data structure: sparse (default), dense or computed (no storing).
    
    **Methods**
    
    ``connect_random(P,Q,p[,weight=1[,fixed=False[,seed=None]]])``
        Connects each neuron in ``P`` to each neuron in ``Q`` with independent
        probability ``p`` and weight ``weight`` (this is the amount that
        gets added to the target state variable). If ``fixed`` is True, then
        the number of presynaptic neurons per neuron is constant. If ``seed``
        is given, it is used as the seed to the random number generators, for
        exactly repeatable results.
    ``connect_full(P,Q[,weight=1])``
        Connect every neuron in ``P`` to every neuron in ``Q`` with the given
        weight.
    ``connect_one_to_one(P,Q)``
        If ``P`` and ``Q`` have the same number of neurons then neuron ``i``
        in ``P`` will be connected to neuron ``i`` in ``Q`` with weight 1.
    ``connect(P,Q,W)``
        You can specify a matrix of weights directly (can be in any format
        recognised by NumPy). Note that due to internal implementation details,
        passing a full matrix rather than a sparse one may slow down your code
        (because zeros will be propagated as well as nonzero values).
        **WARNING:** No unit checking is done at the moment.

    Additionally, you can directly access the matrix of weights by writing::
    
        C = Connection(P,Q)
        print C[i,j]
        C[i,j] = ...
    
    Where here ``i`` is the source neuron and ``j`` is the target neuron.
    Note: if ``C[i,j]`` should be zero, it is more efficient not to write
    ``C[i,j]=0``, if you write this then when neuron ``i`` fires all the
    targets will have the value 0 added to them rather than just the
    nonzero ones.
    **WARNING:** No unit checking is currently done if you use this method.
    Take care to set the right units.
    
    **Advanced information**
    
    The following methods are also defined and used internally, if you are
    writing your own derived connection class you need to understand what
    these do.
    
    ``propagate(spikes)``
        Action to take when source neurons with indices in ``spikes``
        fired.
    ``do_propagate()``
        The method called by the :class:`Network` ``update()`` step,
        typically just propagates the spikes obtained by calling
        the ``get_spikes`` method of the ``source`` :class:`NeuronGroup`.
    '''
    @check_units(delay=second)
    def __init__(self,source,target,state=0,delay=0*msecond,modulation=None,
                 structure='sparse',**kwds):
        self.source=source # pointer to source group
        self.target=target # pointer to target group
        if type(state)==types.StringType: # named state variable
            self.nstate=target.get_var_index(state)
        else:
            self.nstate=state # target state index
        if type(modulation)==types.StringType: # named state variable
            self._nstate_mod=source.get_var_index(modulation)
        else:
            self._nstate_mod=modulation # source state index
        if isinstance(structure,str):
            structure = {'sparse':SparseConnectionMatrix,
                'dense':DenseConnectionMatrix,
                'computed':ComputedConnectionMatrix
                }[structure]
        self.W=structure((len(source),len(target)),**kwds)
        self.iscompressed=False # True if compress() has been called
        source.set_max_delay(delay)
        self.delay=int(delay/source.clock.dt) # Synaptic delay in time bins
#        if self.delay>source._max_delay:
#            raise AttributeError,"Transmission delay is too long."
        
    def reinit(self):
        '''
        Resets the variables.
        '''
        pass
        
    def propagate(self,spikes):
        '''
        Propagates the spikes to the target.
        '''
        #-- Version 1 --
        #for i in spikes.flat:
        #    self.target._S[self.nstate,:]+=self.W[i,:]
        #-- Version 2 --
        #for i in spikes.flat:
        #    self.target._S[self.nstate,self.W.rows[i]]+=self.W.data[i]
        #-- Version 3 --
        #N.B.: not faster to move the state vector to init()
        sv=self.target._S[self.nstate]
        if self._nstate_mod is None:
            self.W.add_rows(spikes,sv)
#            for i in spikes:
#                self.W.add_row(i,sv)
        else:
            sv_pre=self.source._S[self._nstate_mod]
            for i,x in izip(spikes,sv_pre[spikes]):
                self.W.add_scaled_row(i,sv,x)
    
    def do_propagate(self):
        self.propagate(self.source.get_spikes(self.delay))
    
    def origin(self,P,Q):
        '''
        Returns the starting coordinate of the given groups in
        the connection matrix W.
        '''
        return (P._origin-self.source._origin,Q._origin-self.target._origin)

    def compress(self):
        '''
        Converts the connection matrix to a faster structure.
        Replaces array of lists (= lil_matrix) by array of arrays.
        N.B.: that's a hack (many methods will probably not work anymore).
        '''
        if not self.iscompressed:
            self.W.freeze()
            self.iscompressed=True

    # TODO: rewrite all the connection functions to work row by row for memory and time efficiency 

    # TODO: change this
    def connect(self,P,Q,W):
        '''
        Connects (sub)groups P and Q with the weight matrix W (any type).
        Internally: inserts W as a submatrix.
        TODO: checks if the submatrix has already been specified.
        '''
        i0,j0=self.origin(P,Q)
        self.W[i0:i0+len(P),j0:j0+len(Q)]=W
        
    def connect_random(self,P,Q,p,weight=1.,fixed=False, seed=None):
        '''
        Connects the neurons in group P to neurons in group Q with probability p,
        with given weight (default 1).
        The weight can be a quantity or a function of i (in P) and j (in Q).
        If ``fixed`` is True, then the number of presynaptic neurons per neuron is constant.
        '''
        if seed is not None:
            numpy.random.seed(seed) # numpy's random number seed
            pyrandom.seed(seed) # Python's random number seed
        if fixed:
            random_matrix_function=random_matrix_fixed_column
        else:
            random_matrix_function=random_matrix
            
        if callable(weight):
            # Check units
            try:
                weight(0,0)+Q._S0[self.nstate]
            except DimensionMismatchError,inst:
                raise DimensionMismatchError("Incorrects unit for the synaptic weights.",*inst._dims)
            self.connect(P,Q,random_matrix_function(len(P),len(Q),p,value=weight))
        else:
            # Check units
            try:
                weight+Q._S0[self.nstate]
            except DimensionMismatchError,inst:
                raise DimensionMismatchError("Incorrects unit for the synaptic weights.",*inst._dims)
            self.connect(P,Q,random_matrix_function(len(P),len(Q),p,value=float(weight)))

    def connect_full(self,P,Q,weight=1.):
        '''
        Connects the neurons in group P to all neurons in group Q,
        with given weight (default 1).
        The weight can be a quantity or a function of i (in P) and j (in Q).
        '''
        # TODO: check units
        if callable(weight):
            # Check units
            try:
                weight(0,0)+Q._S0[self.nstate]
            except DimensionMismatchError,inst:
                raise DimensionMismatchError("Incorrects unit for the synaptic weights.",*inst._dims)
            W=zeros((len(P),len(Q)))
            try:
                weight(0,1.*arange(0,len(Q)))
                failed=False
            except:
                failed= True
            if failed: # vector-based not possible
                log_debug('connections','Cannot build the connection matrix by rows')
                for i in range(len(P)):
                    for j in range(len(Q)):
                        w = float(weight(i,j))
                        #if not is_within_absolute_tolerance(w,0.,effective_zero):
                        W[i,j] = w
            else:
                for i in range(len(P)): # build W row by row
                    #w = weight(i,1.*arange(0,len(Q)))
                    #I = (abs(w)>effective_zero).nonzero()[0]
                    #print w, I, w[I]
                    #W[i,I] = w[I]
                    W[i,:] = weight(i,1.*arange(0,len(Q)))
            self.connect(P,Q,W)
        else:
            try:
                weight+Q._S0[self.nstate]
            except DimensionMismatchError,inst:
                raise DimensionMismatchError("Incorrect unit for the synaptic weights.",*inst._dims)
            self.connect(P,Q,float(weight)*ones((len(P),len(Q))))

    def connect_one_to_one(self,P,Q,weight=1):
        '''
        Connects P[i] to Q[i] with weights 1 (or weight).
        '''
        if (len(P)!=len(Q)):
            raise AttributeError,'The connected (sub)groups must have the same size.'
        # TODO: unit checking
        self.connect(P,Q,float(weight)*eye_lil_matrix(len(P)))
        
    def __getitem__(self,i):
        return self.W.__getitem__(i)

    def __setitem__(self,i,x):
        # TODO: unit checking
        self.W.__setitem__(i,x)

class IdentityConnection(Connection):
    '''
    A connection between two (sub)groups of the same size, connecting
    P[i] to Q[i] with given weight (default 1)
    '''
    @check_units(delay=second)
    def __init__(self,source,target,state=0,weight=1,delay=0*msecond):
        if (len(source)!=len(target)):
            raise AttributeError,'The connected (sub)groups must have the same size.'
        self.source=source # pointer to source group
        self.target=target # pointer to target group
        if type(state)==types.StringType: # named state variable
            self.nstate=target.get_var_index(state)
        else:
            self.nstate=state # target state index
        self.W=float(weight) # weight
        self.delay=int(delay/source.clock.dt) # Synaptic delay in time bins
        if self.delay>source._max_delay:
            raise AttributeError,"Transmission delay is too long."
        
    def propagate(self,spikes):
        '''
        Propagates the spikes to the target.
        '''
        self.target._S[self.nstate,spikes]+=self.W
        
    def compress(self):
        pass
    
class MultiConnection(Connection):
    '''
    A hub for multiple connections with a common source group.
    '''
    def __init__(self,source,connections=[]):
        self.source=source
        self.connections=connections
        self.iscompressed=False
        self.delay=int(connections[0].delay/source.clock.dt) # Assuming identical delays
        
    def propagate(self,spikes):
        '''
        Propagates the spikes to the targets.
        '''
        for C in self.connections:
            C.propagate(spikes)
            
    def compress(self):
        if not self.iscompressed:
            for C in self.connections:
                C.compress()
            self.iscompressed=True

# Generation of matrices
# TODO: vectorise
def random_matrix(n,m,p,value=1.):
    '''
    Generates a sparse random matrix with size (n,m).
    Entries are 1 (or optionnally value) with probability p.
    If value is a function, then that function is called for each
    non zero element as value() or value(i,j).
    '''
    W=sparse.lil_matrix((n,m))
    if callable(value):
        if value.func_code.co_argcount==0: # TODO: should work with partial objects
            for i in xrange(n):
                k=random.binomial(m,p,1)[0]
                W.rows[i]=sample(xrange(m),k)
                W.rows[i].sort()
                W.data[i]=[value() for _ in xrange(k)]
        elif value.func_code.co_argcount==2:
            for i in xrange(n):
                k=random.binomial(m,p,1)[0]
                W.rows[i]=sample(xrange(m),k)
                W.rows[i].sort()
                W.data[i]=[value(i,j) for j in W.rows[i]]            
        else:
            raise AttributeError,"Bad number of arguments in value function (should be 0 or 2)"
    else:
        for i in xrange(n):
            k=random.binomial(m,p,1)[0]
            # Not significantly faster to generate all random numbers in one pass
            # N.B.: the sample method is implemented in Python and it is not in Scipy
            W.rows[i]=sample(xrange(m),k)
            W.rows[i].sort()
            W.data[i]=[value]*k

    return W

def random_matrix_fixed_column(n,m,p,value=1.):
    '''
    Generates a sparse random matrix with size (n,m).
    Entries are 1 (or optionnally value) with probability p.
    The number of non-zero entries by per column is fixed: (int)(p*n)
    If value is a function, then that function is called for each
    non zero element as value() or value(i,j).
    '''
    W=sparse.lil_matrix((n,m))
    k=(int)(p*n)
    for j in xrange(m):
        # N.B.: the sample method is implemented in Python and it is not in Scipy
        for i in sample(xrange(n),k):
            W.rows[i].append(j)
            
    if callable(value):
        if value.func_code.co_argcount==0:
            for i in xrange(n):
                W.data[i]=[value() for _ in xrange(len(W.rows[i]))]
        elif value.func_code.co_argcount==2:
            for i in xrange(n):
                W.data[i]=[value(i,j) for j in W.rows[i]]            
        else:
            raise AttributeError,"Bad number of arguments in value function (should be 0 or 2)"
    else:
        for i in xrange(n):
            W.data[i]=[value]*len(W.rows[i])

    return W

# Generation of matrices row by row
# TODO: vectorise
def random_matrix_row_by_row(n,m,p,value=1.):
    '''
    Generates a sparse random matrix with size (n,m).
    Entries are 1 (or optionnally value) with probability p.
    If value is a function, then that function is called for each
    non zero element as value() or value(i,j).
    '''
    if callable(value):
        if value.func_code.co_argcount==0:
            for i in xrange(n):
                k=random.binomial(m,p,1)[0]
                row = sample(xrange(m),k)
                row.sort()
                yield row, [value() for _ in xrange(k)]
        elif value.func_code.co_argcount==2:
            for i in xrange(n):
                k=random.binomial(m,p,1)[0]
                row = sample(xrange(m),k)
                row.sort()
                yield row, [value(i,j) for j in W.rows[i]]
        else:
            raise AttributeError,"Bad number of arguments in value function (should be 0 or 2)"
    else:
        for i in xrange(n):
            k=random.binomial(m,p,1)[0]
            row = sample(xrange(m),k)
            row.sort()
            yield row, value 


def eye_lil_matrix(n):
    '''
    Returns the identity matrix of size n as a lil_matrix
    (sparse matrix).
    '''
    M=sparse.lil_matrix((n,n))
    M.setdiag([1.]*n)
    return M

def _define_and_test_interface(self):
    '''
    :class:`Connection`
    ~~~~~~~~~~~~~~~~~~~
    
    **Initialised as:** ::
    
        Connection(source, target[, state=0[, delay=0*ms]])
    
    With arguments:
    
    ``source``
        The group from which spikes will be propagated.
    ``target``
        The group to which spikes will be propagated.
    ``state``
        The state variable name or number that spikes will be
        propagated to in the target group.
    ``delay``
        The delay between a spike being generated at the source
        and received at the target. At the moment, the mechanism
        for delays only works for relatively short delays (an
        error will be generated for delays that are too long), but
        this is subject to change. The exact behaviour then is
        not part of the assured interface, although it is very
        likely that the syntax will not change (or will at least
        be backwards compatible).
    
    **Methods**
    
    ``connect_random(P,Q,p[,weight=1])``
        Connects each neuron in ``P`` to each neuron in ``Q``.
    ``connect_full(P,Q[,weight=1])``
        Connect every neuron in ``P`` to every neuron in ``Q``.
    ``connect_one_to_one(P,Q)``
        If ``P`` and ``Q`` have the same number of neurons then neuron ``i``
        in ``P`` will be connected to neuron ``i`` in ``Q`` with weight 1.
    
    Additionally, you can directly access the matrix of weights by writing::
    
        C = Connection(P,Q)
        print C[i,j]
        C[i,j] = ...
    
    Where here ``i`` is the source neuron and ``j`` is the target neuron.
    Note: No unit checking is currently done if you use this method,
    but this is subject to change for future releases.

    The behaviour when a list of neuron ``spikes`` is received is to
    add ``W[i,:]`` to the target state variable for each ``i`` in ``spikes``. 
    '''
    
    from directcontrol import SpikeGeneratorGroup
    from neurongroup import NeuronGroup
    from network import Network
    from utils.approximatecomparisons import is_approx_equal
    from clock import reinit_default_clock
    
    # test Connection object
    
    eqs = '''
    da/dt = 0.*hertz : 1.
    db/dt = 0.*hertz : 1.
    '''
    
    spikes = [(0,1*msecond),(1,3*msecond)]
    
    G1 = SpikeGeneratorGroup(2,spikes)
    G2 = NeuronGroup(2,model=eqs,threshold=10.,reset=0.)
    
    # first test the methods
    # connect_full
    C = Connection(G1,G2)
    C.connect_full(G1, G2, weight=2.)
    for i in range(2):
        for j in range(2):
            self.assert_(is_approx_equal(C[i,j],2.))
    # connect_random
    C = Connection(G1,G2)
    C.connect_random(G1, G2, 0.5, weight=2.)
    # can't assert anything about that
    # connect_one_to_one
    C = Connection(G1,G2)
    C.connect_one_to_one(G1, G2)
    for i in range(2):
        for j in range(2):
            if i==j:
                self.assert_(is_approx_equal(C[i,j],1.))
            else:
                self.assert_(is_approx_equal(C[i,j],0.))
    del C
    # and we will use a specific set of connections in the next part
    Ca = Connection(G1,G2,'a')
    Cb = Connection(G1,G2,'b')
    Ca[0,0]=1.
    Ca[0,1]=1.
    Ca[1,0]=1.
    #Ca[1,1]=0 by default
    #Cb[0,0]=0 by default
    Cb[0,1]=1.
    Cb[1,0]=1.
    Cb[1,1]=1.
    net = Network(G1,G2,Ca,Cb)
    net.run(2*msecond)
    # after 2 ms, neuron 0 will have fired, so a 0 and 1 should
    # have increased by 1 to [1,1], and b 1 should have increased
    # by 1 to 1
    self.assert_(is_approx_equal(G2.a[0],1.))
    self.assert_(is_approx_equal(G2.a[1],1.))
    self.assert_(is_approx_equal(G2.b[0],0.))
    self.assert_(is_approx_equal(G2.b[1],1.))
    net.run(2*msecond)
    # after 4 ms, neuron 1 will have fired, so a 0 should have
    # increased by 1 to 2, and b 0 and 1 should have increased
    # by 1 to [1, 2]
    self.assert_(is_approx_equal(G2.a[0],2.))
    self.assert_(is_approx_equal(G2.a[1],1.))
    self.assert_(is_approx_equal(G2.b[0],1.))
    self.assert_(is_approx_equal(G2.b[1],2.))
    
    reinit_default_clock()
