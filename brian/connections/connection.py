from base import *
from sparsematrix import *
from connectionvector import *
from constructionmatrix import *
from connectionmatrix import *
from construction import *
from propagation_c_code import *
# we do this at the bottom because of order of import issues
#from delayconnection import * 

__all__ = ['Connection']

class Connection(magic.InstanceTracker, ObjectContainer):
    '''
    Mechanism for propagating spikes from one group to another

    A Connection object declares that when spikes in a source
    group are generated, certain neurons in the target group
    should have a value added to specific states. See
    Tutorial 2: Connections to understand this better.
    
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
        and received at the target. Depending on the type of ``delay``
        it has different effects. If ``delay`` is a scalar value, then
        the connection will be initialised with all neurons having
        that delay. For very long delays, this may raise an error. If
        ``delay=True`` then the connection will be initialised as a
        :class:`DelayConnection`, allowing heterogeneous delays (a
        different delay for each synapse). ``delay`` can also be a
        pair ``(min,max)`` or a function of one or two variables, in
        both cases it will be initialised as a :class:`DelayConnection`,
        see the documentation for that class for details. Note that in
        these cases, initialisation of delays will only have the
        intended effect if used with the ``weight`` and ``sparseness``
        arguments below.
    ``max_delay``
        If you are using a connection with heterogeneous delays, specify
        this to set the maximum allowed delay (smaller values use less
        memory). The default is 5ms.
    ``modulation``
        The state variable name from the source group that scales
        the synaptic weights (for short-term synaptic plasticity).
    ``structure``
        Data structure: ``sparse`` (default), ``dense`` or
        ``dynamic``. See below for more information on structures.
    ``weight``
        If specified, the connection matrix will be initialised with
        values specified by ``weight``, which can be any of the values
        allowed in the methods `connect*`` below.
    ``sparseness``
        If ``weight`` is specified and ``sparseness`` is not, a full
        connection is assumed, otherwise random connectivity with this
        level of sparseness is assumed.
    
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
    
    **Connection matrix structures**
    
    Brian currently features three types of connection matrix structures,
    each of which is suited for different situations. Brian has two stages
    of connection matrix. The first is the construction stage, used for
    building a weight matrix. This stage is optimised for the construction
    of matrices, with lots of features, but would be slow for runtime
    behaviour. Consequently, the second stage is the connection stage,
    used when Brian is being run. The connection stage is optimised for
    run time behaviour, but many features which are useful for construction
    are absent (e.g. the ability to add or remove synapses). Conversion
    between construction and connection stages is done by the
    ``compress()`` method of :class:`Connection` which is called
    automatically when it is used for the first time.
    
    The structures are: 
    
    ``dense``
        A dense matrix. Allows runtime modification of all values. If
        connectivity is close to being dense this is probably the most
        efficient, but in most cases it is less efficient. In addition,
        a dense connection matrix will often do the wrong thing if
        using STDP. Because a synapse will be considered to exist but
        with weight 0, STDP will be able to create new synapses where
        there were previously none. Memory requirements are ``8NM``
        bytes where ``(N,M)`` are the dimensions. (A ``double`` float
        value uses 8 bytes.)
    ``sparse``
        A sparse matrix. See :class:`SparseConnectionMatrix` for
        details on implementation. This class features very fast row
        access, and slower column access if the ``column_access=True``
        keyword is specified (making it suitable for learning
        algorithms such as STDP which require this). Memory
        requirements are 12 bytes per nonzero entry for row access
        only, or 20 bytes per nonzero entry if column access is
        specified. Synapses cannot be created or deleted at runtime
        with this class (although weights can be set to zero).
    ``dynamic``
        A sparse matrix which allows runtime insertion and removal
        of synapses. See :class:`DynamicConnectionMatrix` for
        implementation details. This class features row and column
        access. The row access is slower than for ``sparse`` so this
        class should only be used when insertion and removal of
        synapses is crucial. Memory requirements are 24 bytes per
        nonzero entry. However, note that more memory than this
        may be required because memory is allocated using a
        dynamic array which grows by doubling its size when it runs
        out. If you know the maximum number of nonzero entries you will
        have in advance, specify the ``nnzmax`` keyword to set the
        initial size of the array. 
    
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
    #@check_units(delay=second)
    def __init__(self, source, target, state=0, delay=0 * msecond, modulation=None,
                 structure='sparse', weight=None, sparseness=None, max_delay=5 * ms, **kwds):
        if not isinstance(delay, float):
            if delay is True:
                delay = None # this instructs us to use DelayConnection, but not initialise any delays
            self.__class__ = DelayConnection
            self.__init__(source, target, state=state, modulation=modulation, structure=structure,
                          weight=weight, sparseness=sparseness, delay=delay, max_delay=max_delay, **kwds)
            return
        self.source = source # pointer to source group
        self.target = target # pointer to target group
        if isinstance(state, str): # named state variable
            self.nstate = target.get_var_index(state)
        else:
            self.nstate = state # target state index
        if isinstance(modulation, str): # named state variable
            self._nstate_mod = source.get_var_index(modulation)
        else:
            self._nstate_mod = modulation # source state index
        if isinstance(structure, str):
            structure = construction_matrix_register[structure]
        self.W = structure((len(source), len(target)), **kwds)
        self.iscompressed = False # True if compress() has been called
        source.set_max_delay(delay)
        if not isinstance(self, DelayConnection):
            self.delay = int(delay / source.clock.dt) # Synaptic delay in time bins
        self._useaccel = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
        self._extra_compile_args = ['-O3']
        if self._cpp_compiler == 'gcc':
            self._extra_compile_args += get_global_preference('gcc_options') # ['-march=native', '-ffast-math']
        self._keyword_based_init(weight=weight, sparseness=sparseness)

    def _keyword_based_init(self, weight=None, sparseness=None, **kwds):
        # Initialisation of weights
        # TODO: check consistency of weight and sparseness
        # TODO: select dense or sparse according to sparseness
        if weight is not None or sparseness is not None:
            if weight is None:
                weight = 1.0
            if sparseness is None:
                sparseness = 1
            if isinstance(weight, sparse.lil_matrix) or isinstance(weight, ndarray):
                self.connect(W=weight)
            elif sparseness == 1:
                self.connect_full(weight=weight)
            else:
                self.connect_random(weight=weight, p=sparseness)

    def propagate(self, spikes):
        if not self.iscompressed:
            self.compress()
        if len(spikes):
            # Target state variable
            sv = self.target._S[self.nstate]
            # If specified, modulation state variable
            if self._nstate_mod is not None:
                sv_pre = self.source._S[self._nstate_mod]
            # Get the rows of the connection matrix, each row will be either a
            # DenseConnectionVector or a SparseConnectionVector.
            rows = self.W.get_rows(spikes)
            if not self._useaccel: # Pure Python version is easier to understand, but slower than C++ version below
                if isinstance(rows[0], SparseConnectionVector):
                    if self._nstate_mod is None:
                        # Rows stored as sparse vectors without modulation
                        for row in rows:
                            sv[row.ind] += row
                    else:
                        # Rows stored as sparse vectors with modulation
                        for i, row in izip(spikes, rows):
                            # note we call the numpy __mul__ directly because row is
                            # a SparseConnectionVector with different mul semantics
                            sv[row.ind] += numpy.ndarray.__mul__(row, sv_pre[i])
                else:
                    if self._nstate_mod is None:
                        # Rows stored as dense vectors without modulation
                        for row in rows:
                            sv += row
                    else:
                        # Rows stored as dense vectors with modulation
                        for i, row in izip(spikes, rows):
                            sv += numpy.ndarray.__mul__(row, sv_pre[i])
            else: # C++ accelerated code, does the same as the code above but faster and less pretty
                nspikes = len(spikes)
                if isinstance(rows[0], SparseConnectionVector):
                    rowinds = [r.ind for r in rows]
                    datas = rows
                    if self._nstate_mod is None:
                        code = propagate_weave_code_sparse
                        codevars = propagate_weave_code_sparse_vars
                    else:
                        code = propagate_weave_code_sparse_modulation
                        codevars = propagate_weave_code_sparse_modulation_vars
                else:
                    if not isinstance(spikes, numpy.ndarray):
                        spikes = array(spikes, dtype=int)
                    N = len(sv)
                    if self._nstate_mod is None:
                        code = propagate_weave_code_dense
                        codevars = propagate_weave_code_dense_vars
                    else:
                        code = propagate_weave_code_dense_modulation
                        codevars = propagate_weave_code_dense_modulation_vars
                weave.inline(code, codevars,
                             compiler=self._cpp_compiler,
                             #type_converters=weave.converters.blitz,
                             extra_compile_args=self._extra_compile_args)

    def compress(self):
        if not self.iscompressed:
            self.W = self.W.connection_matrix()
            self.iscompressed = True

    def reinit(self):
        '''
        Resets the variables.
        '''
        pass

    def do_propagate(self):
        self.propagate(self.source.get_spikes(self.delay))

    def origin(self, P, Q):
        '''
        Returns the starting coordinate of the given groups in
        the connection matrix W.
        '''
        return (P._origin - self.source._origin, Q._origin - self.target._origin)

    # TODO: rewrite all the connection functions to work row by row for memory and time efficiency 

    # TODO: change this
    def connect(self, source=None, target=None, W=None):
        '''
        Connects (sub)groups P and Q with the weight matrix W (any type).
        Internally: inserts W as a submatrix.
        TODO: checks if the submatrix has already been specified.
        '''
        P = source or self.source
        Q = target or self.target
        i0, j0 = self.origin(P, Q)
        self.W[i0:i0 + len(P), j0:j0 + len(Q)] = W

    def connect_random(self, source=None, target=None, p=1., weight=1., fixed=False, seed=None, sparseness=None):
        '''
        Connects the neurons in group P to neurons in group Q with probability p,
        with given weight (default 1).
        The weight can be a quantity or a function of i (in P) and j (in Q).
        If ``fixed`` is True, then the number of presynaptic neurons per neuron is constant.
        '''
        P = source or self.source
        Q = target or self.target
        if sparseness is not None: p = sparseness # synonym
        if seed is not None:
            numpy.random.seed(seed) # numpy's random number seed
            pyrandom.seed(seed) # Python's random number seed
        if fixed:
            random_matrix_function = random_matrix_fixed_column
        else:
            random_matrix_function = random_matrix

        if callable(weight):
            # Check units
            try:
                if weight.func_code.co_argcount == 2:
                    weight(0, 0) + Q._S0[self.nstate]
                else:
                    weight() + Q._S0[self.nstate]
            except DimensionMismatchError, inst:
                raise DimensionMismatchError("Incorrects unit for the synaptic weights.", *inst._dims)
            self.connect(P, Q, random_matrix_function(len(P), len(Q), p, value=weight))
        else:
            # Check units
            try:
                weight + Q._S0[self.nstate]
            except DimensionMismatchError, inst:
                raise DimensionMismatchError("Incorrects unit for the synaptic weights.", *inst._dims)
            self.connect(P, Q, random_matrix_function(len(P), len(Q), p, value=float(weight)))

    def connect_full(self, source=None, target=None, weight=1.):
        '''
        Connects the neurons in group P to all neurons in group Q,
        with given weight (default 1).
        The weight can be a quantity or a function of i (in P) and j (in Q).
        '''
        P = source or self.source
        Q = target or self.target
        # TODO: check units
        if callable(weight):
            # Check units
            try:
                weight(0, 0) + Q._S0[self.nstate]
            except DimensionMismatchError, inst:
                raise DimensionMismatchError("Incorrects unit for the synaptic weights.", *inst._dims)
            W = zeros((len(P), len(Q)))
            try:
                x = weight(0, 1. * arange(0, len(Q)))
                failed = False
                if array(x).size != len(Q):
                    failed = True
            except:
                failed = True
            if failed: # vector-based not possible
                log_debug('connections', 'Cannot build the connection matrix by rows')
                for i in range(len(P)):
                    for j in range(len(Q)):
                        w = float(weight(i, j))
                        #if not is_within_absolute_tolerance(w,0.,effective_zero): # for sparse matrices
                        W[i, j] = w
            else:
                for i in range(len(P)): # build W row by row
                    #Below: for sparse matrices (?)
                    #w = weight(i,1.*arange(0,len(Q)))
                    #I = (abs(w)>effective_zero).nonzero()[0]
                    #print w, I, w[I]
                    #W[i,I] = w[I]
                    W[i, :] = weight(i, 1. * arange(0, len(Q)))
            self.connect(P, Q, W)
        else:
            try:
                weight + Q._S0[self.nstate]
            except DimensionMismatchError, inst:
                raise DimensionMismatchError("Incorrect unit for the synaptic weights.", *inst._dims)
            self.connect(P, Q, float(weight) * ones((len(P), len(Q))))

    def connect_one_to_one(self, source=None, target=None, weight=1):
        '''
        Connects source[i] to target[i] with weights 1 (or weight).
        '''
        P = source or self.source
        Q = target or self.target
        if (len(P) != len(Q)):
            raise AttributeError, 'The connected (sub)groups must have the same size.'
        # TODO: unit checking
        self.connect(P, Q, float(weight) * eye_lil_matrix(len(P)))

    def __getitem__(self, i):
        return self.W.__getitem__(i)

    def __setitem__(self, i, x):
        # TODO: unit checking
        self.W.__setitem__(i, x)

from delayconnection import *
