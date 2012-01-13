from base import *
from sparsematrix import *
from connectionvector import *
from constructionmatrix import *
from connectionmatrix import *
from construction import *
from connection import *
from propagation_c_code import *
import warnings
network_operation = None # we import this when we need to because of order of import issues

__all__ = [
         'DelayConnection',
         ]


class DelayConnection(Connection):
    '''
    Connection which implements heterogeneous postsynaptic delays
    
    Initialised as for a :class:`Connection`, but with the additional
    keyword:
    
    ``max_delay``
        Specifies the maximum delay time for any
        neuron. Note, the smaller you make this the less memory will be
        used.
    
    Overrides the following attribute of :class:`Connection`:
    
    .. attribute:: delay
    
        A matrix of delays. This array can be changed during a run,
        but at no point should it be greater than ``max_delay``.
    
    In addition, the methods ``connect``, ``connect_random``, ``connect_full``,
    and ``connect_one_to_one`` have a new keyword ``delay=...`` for setting the
    initial values of the delays, where ``delay`` can be one of:

    * A float, all delays will be set to this value
    * A pair (min, max), delays will be uniform between these two
      values.
    * A function of no arguments, will be called for each nonzero
      entry in the weight matrix.
    * A function of two argument ``(i,j)`` will be called for each
      nonzero entry in the weight matrix.
    * A matrix of an appropriate type (e.g. ndarray or lil_matrix).

    Finally, there is a method:
    
    ``set_delays(source, target, delay)``
        Where ``delay`` must be of one of the types above.
    
    **Notes**
    
    This class implements post-synaptic delays. This means that the spike is
    propagated immediately from the presynaptic neuron with the synaptic
    weight at the time of the spike, but arrives at the postsynaptic neuron
    with the given delay. At the moment, Brian only provides support for
    presynaptic delays if they are homogeneous, using the ``delay`` keyword
    of a standard ``Connection``.
    
    **Implementation**
    
    :class:`DelayConnection` stores an array of size ``(n,m)`` where
    ``n`` is ``max_delay/dt`` for ``dt`` of the target :class:`NeuronGroup`'s clock,
    and ``m`` is the number of neurons in the target. This array can potentially
    be quite large. Each row in this array represents the array that should be
    added to the target state variable at some particular future time. Which
    row corresponds to which time is tracked using a circular indexing scheme.
    
    When a spike from neuron ``i`` in the source is encountered, the delay time
    of neuron ``i`` is looked up, the row corresponding to the current time
    plus that delay time is found using the circular indexing scheme, and then
    the spike is propagated to that row as for a standard connection (although
    this won't be propagated to the target until a later time).
    
    **Warning**
    
    If you are using a dynamic connection matrix, it is your responsibility to
    ensure that the nonzero entries of the weight matrix and the delay matrix
    exactly coincide. This is not an issue for sparse or dense matrices.
    '''

    def __init__(self, source, target, state=0, modulation=None,
                 structure='sparse',
                 weight=None, sparseness=None, delay=None,
                 max_delay=5 * msecond, **kwds):
        global network_operation
        if network_operation is None:
            from ..network import network_operation
        Connection.__init__(self, source, target, state=state, modulation=modulation,
                            structure=structure, weight=weight, sparseness=sparseness, **kwds)
        self._max_delay = int(max_delay / target.clock.dt) + 1
        source.set_max_delay(max_delay)
        # Each row of the following array stores the cumulative effect of spikes at some
        # particular time, defined by a circular indexing scheme. The _cur_delay_ind attribute
        # stores the row corresponding to the current time, so that _cur_delay_ind+1 corresponds
        # to that time + target.clock.dt, and so on. When _cur_delay_ind reaches _max_delay it
        # resets to zero.
        self._delayedreaction = numpy.zeros((self._max_delay, len(target)))
        # vector of delay times, can be changed during a run
        if isinstance(structure, str):
            structure = construction_matrix_register[structure]
        self.delayvec = structure((len(source), len(target)), **kwds)
        self._cur_delay_ind = 0
        # this network operation is added to the Network object via the contained_objects
        # protocol (see the line after the function definition). The standard Connection.propagate
        # function propagates spikes to _delayedreaction rather than the target, and this
        # function which is called after the usual propagations propagates that data from
        # _delayedreaction to the target. It only needs to be called each target.clock update.
        @network_operation(clock=target.clock, when='after_connections')
        def delayed_propagate():
            # propagate from _delayedreaction -> target group
            target._S[self.nstate] += self._delayedreaction[self._cur_delay_ind, :]
            # reset the current row of _delayedreaction
            self._delayedreaction[self._cur_delay_ind, :] = 0.0
            # increase the index for the circular indexing scheme
            self._cur_delay_ind = (self._cur_delay_ind + 1) % self._max_delay
        self.delayed_propagate = delayed_propagate
        self.contained_objects = [delayed_propagate]
        # this is just used to convert delayvec's which are in ms to integers, precalculating it makes it faster
        self._invtargetdt = 1 / self.target.clock._dt
        self._useaccel = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
        self._extra_compile_args = ['-O3']
        if self._cpp_compiler == 'gcc':
            self._extra_compile_args += get_global_preference('gcc_options') # ['-march=native', '-ffast-math']
        if delay is not None:
            self.set_delays(delay=delay)

    def propagate(self, spikes):
        if not self.iscompressed:
            self.compress()
        if len(spikes):
            # Target state variable
            dr = self._delayedreaction
            # If specified, modulation state variable
            if self._nstate_mod is not None:
                sv_pre = self.source._S[self._nstate_mod]
            # Get the rows of the connection matrix, each row will be either a
            # DenseConnectionVector or a SparseConnectionVector.
            rows = self.W.get_rows(spikes)
            dvecrows = self.delayvec.get_rows(spikes)
            if not self._useaccel: # Pure Python version is easier to understand, but slower than C++ version below
                if isinstance(rows[0], SparseConnectionVector):
                    if self._nstate_mod is None:
                        # Rows stored as sparse vectors without modulation
                        for row, dvecrow in izip(rows, dvecrows):
                            if not len(row.ind) == len(dvecrow.ind):
                                raise RuntimeError('Weight and delay matrices must be kept in synchrony for sparse matrices.')
                            drind = (self._cur_delay_ind + numpy.array(self._invtargetdt * dvecrow, dtype=int)) % self._max_delay
                            dr[drind, dvecrow.ind] += row
                    else:
                        # Rows stored as sparse vectors with modulation
                        for i, row, dvecrow in izip(spikes, rows, dvecrows):
                            if not len(row.ind) == len(dvecrow.ind):
                                raise RuntimeError('Weight and delay matrices must be kept in synchrony for sparse matrices.')
                            drind = (self._cur_delay_ind + numpy.array(self._invtargetdt * dvecrow, dtype=int)) % self._max_delay
                            # note we call the numpy __mul__ directly because row is
                            # a SparseConnectionVector with different mul semantics
                            dr[drind, dvecrow.ind] += numpy.ndarray.__mul__(row, sv_pre[i])
                else:
                    if self._nstate_mod is None:
                        # Rows stored as dense vectors without modulation
                        drjind = numpy.arange(len(self.target), dtype=int)
                        for row, dvecrow in izip(rows, dvecrows):
                            drind = (self._cur_delay_ind + numpy.array(self._invtargetdt * dvecrow, dtype=int)) % self._max_delay
                            dr[drind, drjind[:len(drind)]] += row
                    else:
                        # Rows stored as dense vectors with modulation
                        drjind = numpy.arange(len(self.target), dtype=int)
                        for i, row, dvecrow in izip(spikes, rows, dvecrows):
                            drind = (self._cur_delay_ind + numpy.array(self._invtargetdt * dvecrow, dtype=int)) % self._max_delay
                            dr[drind, drjind[:len(drind)]] += numpy.ndarray.__mul__(row, sv_pre[i])
            else: # C++ accelerated code, does the same as the code above but faster and less pretty
                nspikes = len(spikes)
                cdi = self._cur_delay_ind
                idt = self._invtargetdt
                md = self._max_delay
                if isinstance(rows[0], SparseConnectionVector):
                    rowinds = [r.ind for r in rows]
                    datas = rows
                    if self._nstate_mod is None:
                        code = delay_propagate_weave_code_sparse
                        codevars = delay_propagate_weave_code_sparse_vars
                    else:
                        code = delay_propagate_weave_code_sparse_modulation
                        codevars = delay_propagate_weave_code_sparse_modulation_vars
                else:
                    if not isinstance(spikes, numpy.ndarray):
                        spikes = array(spikes, dtype=int)
                    N = len(self.target)
                    if self._nstate_mod is None:
                        code = delay_propagate_weave_code_dense
                        codevars = delay_propagate_weave_code_dense_vars
                    else:
                        code = delay_propagate_weave_code_dense_modulation
                        codevars = delay_propagate_weave_code_dense_modulation_vars
                weave.inline(code, codevars,
                             compiler=self._cpp_compiler,
                             type_converters=weave.converters.blitz,
                             extra_compile_args=self._extra_compile_args)

    def do_propagate(self):
        self.propagate(self.source.get_spikes(0))

    def _set_delay_property(self, val):
        self.delayvec[:] = val

    delay = property(fget=lambda self:self.delayvec, fset=_set_delay_property)

    def compress(self):
        if not self.iscompressed:
            # We want delayvec to have nonzero entries at the same places as
            # W does, so we use W to initialise the compressed version of
            # delayvec, and then copy the values from the old delayvec to
            # the new compressed one, allowing delayvec and W to not have
            # to be perfectly intersected at the initialisation stage. If
            # the structure is dynamic, it will be the user's
            # responsibility to update them in sequence
            delayvec = self.delayvec
            # Special optimisation for the lil_matrix case if the indices
            # already match. Note that this special optimisation also allows
            # you to use multiple synapses via a sort of hack (editing the
            # rows and data attributes of the lil_matrix explicitly)
            using_lil_matrix = False
            if isinstance(delayvec, sparse.lil_matrix):
                using_lil_matrix = True
            self.delayvec = self.W.connection_matrix(copy=True)
            repeated_index_hack = False
            for i in xrange(self.W.shape[0]):
                if using_lil_matrix:
                    try:
                        row = SparseConnectionVector(self.W.shape[1],
                                                     array(delayvec.rows[i], dtype=int),
                                                     array(delayvec.data[i]))
                        if sum(diff(row.ind)==0)>0:
                            warnings.warn("You are using the repeated index hack, be careful!")
                            repeated_index_hack = True
                        self.delayvec.set_row(i, row)
                    except ValueError:
                        if repeated_index_hack:
                            warnings.warn("You are using the repeated index "
                                          "hack and not ensuring that weight "
                                          "and delay indices correspond, this "
                                          "is almost certainly an error!")
                        using_lil_matrix = False
                        # The delayvec[i,:] operation for sparse.lil_matrix format
                        # is VERY slow, but the CSR format is fine.
                        delayvec = delayvec.tocsr()
                if not using_lil_matrix:
                    self.delayvec.set_row(i, array(todense(delayvec[i, :]), copy=False).flatten())
                
            Connection.compress(self)

    def set_delays(self, source=None, target=None, delay=None):
        '''
        Set the delays corresponding to the weight matrix
        
        ``delay`` must be one of:
        
        * A float, all delays will be set to this value
        * A pair (min, max), delays will be uniform between these two
          values.
        * A function of no arguments, will be called for each nonzero
          entry in the weight matrix.
        * A function of two argument ``(i,j)`` will be called for each
          nonzero entry in the weight matrix.
        * A matrix of an appropriate type (e.g. ndarray or lil_matrix).
        '''
        if delay is None:
            return
        W = self.W
        P = source or self.source
        Q = target or self.target
        i0, j0 = self.origin(P, Q)
        i1 = i0 + len(P)
        j1 = j0 + len(Q)
        if isinstance(W, sparse.lil_matrix):
            def getrow(i):
                inds = array(W.rows[i], dtype=int)
                inds = inds[logical_and(inds >= j0, inds < j1)]
                return inds, len(inds)
        else:
            def getrow(i):
                inds = (W[i, j0:j1] != 0).nonzero()[0] + j0
                return inds, len(inds)
                #return slice(j0, j1), j1-j0
        if isinstance(delay, (float, int)):
            for i in xrange(i0, i1):
                inds, L = getrow(i)
                self.delayvec[i, inds] = delay
        elif isinstance(delay, (tuple, list)) and len(delay) == 2:
            delaymin, delaymax = delay
            for i in xrange(i0, i1):
                inds, L = getrow(i)
                rowdelay = rand(L) * (delaymax - delaymin) + delaymin
                self.delayvec[i, inds] = rowdelay
        elif callable(delay) and delay.func_code.co_argcount == 0:
            for i in xrange(i0, i1):
                inds, L = getrow(i)
                rowdelay = [delay() for _ in xrange(L)]
                self.delayvec[i, inds] = rowdelay
        elif callable(delay) and delay.func_code.co_argcount == 2:
            for i in xrange(i0, i1):
                inds, L = getrow(i)
                if isinstance(inds, slice):
                    inds = numpy.arange(inds.start, inds.stop)
                self.delayvec[i, inds] = delay(i - i0, inds - j0)
        else:
            #raise TypeError('delays must be float, pair or function of 0 or 2 arguments')
            self.delayvec[i0:i1, j0:j1] = delay # probably won't work, but then it will raise an error

    def connect(self, source=None, target=None, W=None, delay=None):
        Connection.connect(self, source=source, target=target, W=W)
        if delay is not None:
            self.set_delays(source, target, delay)

    def connect_random(self, source=None, target=None, p=1.0, weight=1.0,
                       fixed=False, seed=None, sparseness=None, delay=None):
        Connection.connect_random(self, source=source, target=target, p=p,
                                  weight=weight, fixed=fixed, seed=seed,
                                  sparseness=sparseness)
        if delay is not None:
            self.set_delays(source, target, delay)

    def connect_full(self, source=None, target=None, weight=1.0, delay=None):
        Connection.connect_full(self, source=source, target=target, weight=weight)
        if delay is not None:
            self.set_delays(source, target, delay)

    def connect_one_to_one(self, source=None, target=None, weight=1.0, delay=None):
        Connection.connect_one_to_one(self, source=source, target=target,
                                      weight=weight)
        if delay is not None:
            self.set_delays(source, target, delay)
