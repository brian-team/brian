'''
Ready to go into the main part of Brian
'''
from brian import *
from brian.connection import construction_matrix_register, DenseConnectionVector, SparseConnectionVector, todense
from scipy import weave
from scipy import sparse
from operator import isNumberType
from itertools import izip
import numpy

__all__ = ['DelayConnection']

class DelayConnection(Connection):
    '''
    Connection which implements heterogeneous postsynaptic delays
    
    Initialised as for a :class:`Connection`, but with the additional
    keyword:
    
    ``max_delay``
        Specifies the maximum delay time for any
        neuron. Note, the smaller you make this the less memory will be
        used.
    
    
    Has one attribute other than those in a :class:`Connection`:
    
    .. attribute:: delays
    
        A matrix of delays. This array can be changed during a run,
        but at no point should it be greater than ``max_delay``.
    
    In addition, the methods ``connect``, ``connect_random``, ``connect_full``,
    and ``connect_one_to_one`` have a new keyword ``delays=...`` for setting the
    initial values of the delays, where ``delays`` can be one of:

    * A float, all delays will be set to this value
    * A pair (min, max), delays will be uniform between these two
      values.
    * A function of no arguments, will be called for each nonzero
      entry in the weight matrix.
    * A function of two argument ``(i,j)`` will be called for each
      nonzero entry in the weight matrix.
    * A matrix of an appropriate type (e.g. ndarray or lil_matrix).

    Finally, there is a method:
    
    ``set_delays(delays)``
        Where ``delays`` must be of one of the types above.
    
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
                 structure='sparse', max_delay=5*msecond, **kwds):
        Connection.__init__(self, source, target, state=state, modulation=modulation, structure=structure, **kwds)
        self._max_delay = int(max_delay/target.clock.dt)+1
        # Each row of the following array stores the cumulative effect of spikes at some
        # particular time, defined by a circular indexing scheme. The _cur_delay_ind attribute
        # stores the row corresponding to the current time, so that _cur_delay_ind+1 corresponds
        # to that time + target.clock.dt, and so on. When _cur_delay_ind reaches _max_delay it
        # resets to zero.
        self._delayedreaction = numpy.zeros((self._max_delay, len(target)))
        # vector of delay times, can be changed during a run
        if isinstance(structure,str):
            structure = construction_matrix_register[structure]
        self.delayvec = structure((len(source),len(target)),**kwds)
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
        self.contained_objects = [delayed_propagate]
        # this is just used to convert delayvec's which are in ms to integers, precalculating it makes it faster
        self._invtargetdt = 1/self.target.clock._dt
        self._useaccel = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
        
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
                            if not len(row.ind)==len(dvecrow.ind):
                                raise RuntimeError('Weight and delay matrices must be kept in synchrony for sparse matrices.')
                            drind = (self._cur_delay_ind+numpy.array(self._invtargetdt*dvecrow, dtype=int))%self._max_delay
                            dr[drind, dvecrow.ind] += row
                    else:
                        # Rows stored as sparse vectors with modulation
                        for i, row, dvecrow in izip(spikes, rows, dvecrows):
                            if not len(row.ind)==len(dvecrow.ind):
                                raise RuntimeError('Weight and delay matrices must be kept in synchrony for sparse matrices.')
                            drind = (self._cur_delay_ind+numpy.array(self._invtargetdt*dvecrow, dtype=int))%self._max_delay
                            # note we call the numpy __mul__ directly because row is
                            # a SparseConnectionVector with different mul semantics
                            dr[drind, dvecrow.ind] += numpy.ndarray.__mul__(row, sv_pre[i])
                else:
                    if self._nstate_mod is None:
                        # Rows stored as dense vectors without modulation
                        drjind = numpy.arange(len(self.target), dtype=int)
                        for row, dvecrow in izip(rows, dvecrows):
                            drind = (self._cur_delay_ind+numpy.array(self._invtargetdt*dvecrow, dtype=int))%self._max_delay
                            dr[drind, drjind[:len(drind)]] += row
                    else:
                        # Rows stored as dense vectors with modulation
                        drjind = numpy.arange(len(self.target), dtype=int)
                        for i, row, dvecrow in izip(spikes, rows, dvecrows):
                            drind = (self._cur_delay_ind+numpy.array(self._invtargetdt*dvecrow, dtype=int))%self._max_delay
                            dr[drind, drjind[:len(drind)]] += numpy.ndarray.__mul__(row, sv_pre[i])                     
            else: # C++ accelerated code, does the same as the code above but faster and less pretty
                if isinstance(rows[0], SparseConnectionVector):
                    if self._nstate_mod is None:
                        nspikes = len(spikes)
                        rowinds = [r.ind for r in rows]
                        datas = rows
                        cdi = self._cur_delay_ind
                        idt = self._invtargetdt
                        md = self._max_delay
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyObject* _rowind = rowinds[j];
                                    PyArrayObject* _row = convert_to_numpy(_rowind, "row");
                                    conversion_numpy_check_type(_row, PyArray_INT, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<int,1> row = convert_to_blitz<int,1>(_row,"row");
                                    PyObject* _datasj = datas[j];
                                    PyArrayObject* _data = convert_to_numpy(_datasj, "data");
                                    conversion_numpy_check_type(_data, PyArray_DOUBLE, "data");
                                    conversion_numpy_check_size(_data, 1, "data");
                                    blitz::Array<double,1> data = convert_to_blitz<double,1>(_data,"data");
                                    PyObject* _dvecrowsj = dvecrows[j];
                                    PyArrayObject* _dvecrow = convert_to_numpy(_dvecrowsj, "dvecrow");
                                    conversion_numpy_check_type(_dvecrow, PyArray_DOUBLE, "dvecrow");
                                    conversion_numpy_check_size(_dvecrow, 1, "dvecrow");
                                    blitz::Array<double,1> dvecrow = convert_to_blitz<double,1>(_dvecrow,"dvecrow");
                                    int m = row.numElements();
                                    for(int k=0;k<m;k++)
                                    {
                                        dr((cdi+(int)(idt*dvecrow(k)))%md, row(k)) += data(k);
                                    }
                                    Py_DECREF(_rowind);
                                    Py_DECREF(_datasj);
                                    Py_DECREF(_dvecrowsj);
                                }
                                """
                        weave.inline(code,['rowinds','datas','spikes','nspikes',
                                           'dvecrows', 'dr', 'cdi', 'idt', 'md'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])
                    else:
                        nspikes = len(spikes)
                        rowinds = [r.ind for r in rows]
                        datas = rows
                        cdi = self._cur_delay_ind
                        idt = self._invtargetdt
                        md = self._max_delay
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyObject* _rowind = rowinds[j];
                                    PyArrayObject* _row = convert_to_numpy(_rowind, "row");
                                    conversion_numpy_check_type(_row, PyArray_INT, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<int,1> row = convert_to_blitz<int,1>(_row,"row");
                                    PyObject* _datasj = datas[j];
                                    PyArrayObject* _data = convert_to_numpy(_datasj, "data");
                                    conversion_numpy_check_type(_data, PyArray_DOUBLE, "data");
                                    conversion_numpy_check_size(_data, 1, "data");
                                    blitz::Array<double,1> data = convert_to_blitz<double,1>(_data,"data");
                                    PyObject* _dvecrowsj = dvecrows[j];
                                    PyArrayObject* _dvecrow = convert_to_numpy(_dvecrowsj, "dvecrow");
                                    conversion_numpy_check_type(_dvecrow, PyArray_DOUBLE, "dvecrow");
                                    conversion_numpy_check_size(_dvecrow, 1, "dvecrow");
                                    blitz::Array<double,1> dvecrow = convert_to_blitz<double,1>(_dvecrow,"dvecrow");
                                    int m = row.numElements();
                                    double mod = sv_pre(spikes(j));
                                    for(int k=0;k<m;k++)
                                    {
                                        dr((cdi+(int)(idt*dvecrow(k)))%md, row(k)) += data(k)*mod;
                                    }
                                    Py_DECREF(_rowind);
                                    Py_DECREF(_datasj);
                                    Py_DECREF(_dvecrowsj);
                                }
                                """
                        weave.inline(code,['sv_pre','rowinds','datas','spikes','nspikes',
                                           'dvecrows', 'dr', 'cdi', 'idt', 'md'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])
                else:
                    if self._nstate_mod is None:
                        if not isinstance(spikes, numpy.ndarray):
                            spikes = array(spikes, dtype=int)
                        nspikes = len(spikes)
                        N = len(self.target)
                        cdi = self._cur_delay_ind
                        idt = self._invtargetdt
                        md = self._max_delay
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyObject* _rowsj = rows[j];
                                    PyArrayObject* _row = convert_to_numpy(_rowsj, "row");
                                    conversion_numpy_check_type(_row, PyArray_DOUBLE, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<double,1> row = convert_to_blitz<double,1>(_row,"row");
                                    PyObject* _dvecrowsj = dvecrows[j];
                                    PyArrayObject* _dvecrow = convert_to_numpy(_dvecrowsj, "dvecrow");
                                    conversion_numpy_check_type(_dvecrow, PyArray_DOUBLE, "dvecrow");
                                    conversion_numpy_check_size(_dvecrow, 1, "dvecrow");
                                    blitz::Array<double,1> dvecrow = convert_to_blitz<double,1>(_dvecrow,"dvecrow");
                                    for(int k=0;k<N;k++)
                                        dr((cdi+(int)(idt*dvecrow(k)))%md, k) += row(k);
                                    Py_DECREF(_rowsj);
                                    Py_DECREF(_dvecrowsj);
                                }
                                """
                        weave.inline(code,['spikes','nspikes','N', 'rows',
                                           'dr','cdi','idt','md','dvecrows'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])
                    else:
                        if not isinstance(spikes, numpy.ndarray):
                            spikes = array(spikes, dtype=int)
                        nspikes = len(spikes)
                        N = len(self.target)
                        cdi = self._cur_delay_ind
                        idt = self._invtargetdt
                        md = self._max_delay
                        code =  """
                                for(int j=0;j<nspikes;j++)
                                {
                                    PyObject* _rowsj = rows[j];
                                    PyArrayObject* _row = convert_to_numpy(_rowsj, "row");
                                    conversion_numpy_check_type(_row, PyArray_DOUBLE, "row");
                                    conversion_numpy_check_size(_row, 1, "row");
                                    blitz::Array<double,1> row = convert_to_blitz<double,1>(_row,"row");
                                    PyObject* _dvecrowsj = dvecrows[j];
                                    PyArrayObject* _dvecrow = convert_to_numpy(_dvecrowsj, "dvecrow");
                                    conversion_numpy_check_type(_dvecrow, PyArray_DOUBLE, "dvecrow");
                                    conversion_numpy_check_size(_dvecrow, 1, "dvecrow");
                                    blitz::Array<double,1> dvecrow = convert_to_blitz<double,1>(_dvecrow,"dvecrow");
                                    double mod = sv_pre(spikes(j));
                                    for(int k=0;k<N;k++)
                                        dr((cdi+(int)(idt*dvecrow(k)))%md, k) += row(k)*mod;
                                    Py_DECREF(_rowsj);
                                    Py_DECREF(_dvecrowsj);
                                }
                                """
                        weave.inline(code,['sv_pre','spikes','nspikes','N', 'rows',
                                           'dr','cdi','idt','md','dvecrows'],
                                     compiler=self._cpp_compiler,
                                     type_converters=weave.converters.blitz,
                                     extra_compile_args=['-O3'])
            
    def _set_delays_property(self, val):
        self.delayvec[:]=val

    delays = property(fget=lambda self:self.delayvec, fset=_set_delays_property)

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
            self.delayvec = self.W.connection_matrix()
            for i in xrange(self.W.shape[0]):
                self.delayvec.set_row(i, array(todense(delayvec[i,:]), copy=False).flatten())    
            Connection.compress(self)
    
    def set_delays(self, delays):
        '''
        Set the delays corresponding to the weight matrix
        
        ``delays`` must be one of:
        
        * A float, all delays will be set to this value
        * A pair (min, max), delays will be uniform between these two
          values.
        * A function of no arguments, will be called for each nonzero
          entry in the weight matrix.
        * A function of two argument ``(i,j)`` will be called for each
          nonzero entry in the weight matrix.
        * A matrix of an appropriate type (e.g. ndarray or lil_matrix).
        '''
        W = self.W
        if isinstance(W, sparse.lil_matrix):
            def getrow(i):
                return W.rows[i], W.data[i]
        else:
            def getrow(i):
                return slice(None), W[i,:]
        if isinstance(delays, float):
            for i in xrange(self.W.shape[0]):
                inds, data = getrow(i)
                self.delayvec[i, inds] = delays
        elif isinstance(delays, (tuple, list)) and len(delays)==2:
            delaymin, delaymax = delays
            for i in xrange(self.W.shape[0]):
                inds, data = getrow(i)
                rowdelay = rand(len(data))
                self.delayvec[i, inds] = rowdelay
        elif callable(delays) and delays.func_code.co_argcount==0:
            for i in xrange(self.W.shape[0]):
                inds, data = getrow(i)
                rowdelay = [delays() for _ in xrange(len(data))]
                self.delayvec[i, inds] = rowdelay
        elif callable(delays) and delays.func_code.co_argcount==2:
            for i in xrange(self.W.shape[0]):
                inds, data = getrow(i)
                if inds==slice(None):
                    inds = arange(len(data))
                self.delayvec[i, inds] = delays(i, inds)
        else:
            #raise TypeError('delays must be float, pair or function of 0 or 2 arguments')
            self.delayvec[:,:] = delays # probably won't work, but then it will raise an error

    def connect(self, *args, **kwds):
        delays = kwds.pop('delays', None)
        Connection.connect(self, *args, **kwds)
        if delays is not None:
            self.set_delays(delays)

    def connect_random(self, *args, **kwds):
        delays = kwds.pop('delays', None)
        Connection.connect_random(self, *args, **kwds)
        if delays is not None:
            self.set_delays(delays)
    
    def connect_full(self, *args, **kwds):
        delays = kwds.pop('delays', None)
        Connection.connect_full(self, *args, **kwds)
        if delays is not None:
            self.set_delays(delays)

    def connect_one_to_one(self, *args, **kwds):
        delays = kwds.pop('delays', None)
        Connection.connect_one_to_one(self, *args, **kwds)
        if delays is not None:
            self.set_delays(delays)

if __name__=='__main__':
    
    maxdelay = 5*ms
    N = 100
    M = 100
    
    #inp = SpikeGeneratorGroup(N, [(i, 0*ms) for i in range(N)])
    inp = NeuronGroup(N, model='V:1\nW:1', threshold=-1.0, reset=-2.0)
    inp.W[:50] = 1
    inp.W[50:] = 0
    outp = NeuronGroup(M, model='V:1', threshold=0.5, reset=0.0)
    
    C = DelayConnection(inp, outp, structure='dense', modulation='W')
    #C.connect_full(inp, outp, delays=(0*ms, 4*ms))
    C.connect_one_to_one(inp, outp, delays=lambda:clip(0.2*ms*randn()+2*ms, 0*ms, 4*ms))
#    for i in xrange(N):
#        for j in xrange(M):
#            C.delays[i,j] = (maxdelay*i*j)/((N-1)*(M-1))
#    C.delays[0,1] = 3*ms
#    C.delays[0,2] = 2*ms
    #C.set_delays(lambda:clip(0.2*ms*randn()+2*ms, 0*ms, 4*ms))
    #C.set_delays(lambda i, j:4*ms*clip(i/float(N),0,(j/float(M))**2))
            
    M_outp = SpikeMonitor(outp)
    
    run(20*ms)
    
#    print M_outp.spikes
    
    raster_plot()
    show()
