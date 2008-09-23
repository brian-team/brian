from brian import *
from brian.connection import SparseConnectionMatrix, DenseConnectionMatrix, ComputedConnectionMatrix
from scipy import weave
from operator import isNumberType
import numpy
try:
    from stdp_sparse import SparseSTDPConnectionMatrix
except ImportError:
    class SparseSTDPConnectionMatrix(object):
        pass

__all__ = ['DelayConnection','PresynapticDelayConnection']

class DelayConnection(Connection):
    '''
    Connection which implements heterogeneous postsynaptic delays
    
    Initialised as for a :class:`Connection`, but with the additional
    keywords:
    
    ``max_delay``
        Specifies the maximum delay time for any
        neuron. Note, the smaller you make this the less memory will be
        used.
    ``delay_dimension``
        By default, delays are specified as a 2D array giving the delays
        for every synapse ``i`` to ``j``. By specifying ``delay_dimension=1``
        you can set delays as a 1D array giving the delays for every source
        neuron only. Note though that the delays are implemented
        postsynaptically, so this setting may not always give the expected
        results.
    
    NOTE: At the moment, the ``modulation`` keyword is ignored, as this has
    not yet been implemented.
    
    Has one attribute other than those in a :class:`Connection`:
    
    .. attribute:: delays
    
        An array of delays. If ``delay_dimension=1``, then this is a
        1D array of delays for each neuron in the source group, if ``2``
        then this is a 2D array of delays for each synapse, so that
        ``delays[i,j]`` is the delay from source neuron ``i`` to target
        neuron ``j``. In either case, this array can be changed during a run,
        but at no point should it be greater than ``max_delay``. In the 2D
        case, if ``structure='sparse'`` then ``delays`` is also a sparse
        matrix in the numpy ``lil_matrix`` format, which may not be very
        efficient - if you do not plan on changing the values of ``delays``
        during a run, do this for improved efficiency::
        
            C.delays.freeze()
    
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
    '''
       
    def __init__(self, source, target, state=0, modulation=None,
                 structure='sparse', max_delay=5*msecond, delay_dimension=2, **kwds):
        if modulation is not None:
            raise ValueError('Modulation not yet supported for DelayConnection')
        Connection.__init__(self, source, target, state=state, modulation=modulation, structure=structure, **kwds)
        self._max_delay = int(max_delay/target.clock.dt)+1
        # Each row of the following array stores the cumulative effect of spikes at some
        # particular time, defined by a circular indexing scheme. The _cur_delay_ind attribute
        # stores the row corresponding to the current time, so that _cur_delay_ind+1 corresponds
        # to that time + target.clock.dt, and so on. When _cur_delay_ind reaches _max_delay it
        # resets to zero.
        self._delayedreaction = numpy.zeros((self._max_delay, len(target)))
        # vector of delay times, can be changed during a run
        if delay_dimension==1:
            self.delayvec = numpy.zeros(len(source))
        elif delay_dimension==2:
            if isinstance(structure,str):
                structure = {'sparse':SparseConnectionMatrix,
                    'dense':DenseConnectionMatrix,
                    'computed':ComputedConnectionMatrix
                    }[structure]
            self.delayvec=structure((len(source),len(target)),**kwds)
#            self.delayvec = numpy.zeros((len(source), len(target)))
        else:
            raise ValueError('delay_dimension must be 1 or 2')
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
        # TODO: modulation not yet supported
        if self.delayvec.ndim==1:
            for i in spikes:
                # standard propagation, but to the 'state variable' in _delayedreaction defined by the circular
                # indexing scheme
                sv = self._delayedreaction[(self._cur_delay_ind+int(self._invtargetdt*self.delayvec[i]))%self._max_delay,:]
                self.W.add_row(i,sv)
        elif self.delayvec.ndim==2:
            if False and self._useaccel and isinstance(self.W, SparseSTDPConnectionMatrix) and isinstance(self.delayvec,DenseConnectionMatrix):
                if not isinstance(spikes, numpy.ndarray):
                    spikes = array(spikes, dtype=int)
                rowj = self.W.rowj
                rowdata = self.W.rowdata
                nspikes = len(spikes)
                cdi = self._cur_delay_ind
                delayvec = asarray(self.delayvec)
                dr = self._delayedreaction
                invdt = self._invtargetdt
                maxdelay = self._max_delay
                code =  """
                for(int i=0;i<nspikes;i++)
                {
                    int j = spikes(i);
                    PyArrayObject* _thisrowj = convert_to_numpy(rowj[j], "rowj");
                    conversion_numpy_check_type(_thisrowj, PyArray_INT, "rowj");
                    conversion_numpy_check_size(_thisrowj, 1, "rowj");
                    blitz::Array<int,1> thisrowj = convert_to_blitz<int,1>(_thisrowj,"row");
                    PyArrayObject* _thisrowdata = convert_to_numpy(rowdata[j], "rowdata");
                    conversion_numpy_check_type(_thisrowdata, PyArray_DOUBLE, "rowdata");
                    conversion_numpy_check_size(_thisrowdata, 1, "rowdata");
                    blitz::Array<double,1> thisrowdata = convert_to_blitz<double,1>(_thisrowdata,"rowdata");
                    int m = thisrowj.numElements();
                    for(int k=0;k<m;k++)
                    {
                        int drind = (cdi + (int)(invdt*delayvec(j, thisrowj(k)))) % maxdelay;
                        dr(drind, thisrowj(k)) = dr(drind, thisrowj(k)) + thisrowdata(k);
                    }
                }
                """
                weave.inline(code,['rowj','rowdata','dr','delayvec','invdt','maxdelay','cdi','spikes','nspikes'],
                             compiler=self._cpp_compiler,
                             type_converters=weave.converters.blitz,
                             extra_compile_args=['-O3'])
            else:
                for i in spikes:
                    # TODO: fix this temporary hack until we sort out generalised connection matrices better
                    dvecrow = self.delayvec[i,:]
                    # handle the case where W is a sparse matrix
                    if hasattr(dvecrow,'toarray'):
                        dvecrow = dvecrow.toarray().squeeze()
                    drind = (self._cur_delay_ind+numpy.array(self._invtargetdt*dvecrow,dtype=int))%self._max_delay
                    # TODO: fix this temporary hack until we sort out generalised connection matrices better
                    Wrow = self.W[i,:]
                    # handle the case where W is a sparse matrix
                    if hasattr(Wrow,'toarray'):
                        Wrow = Wrow.toarray().squeeze()
                    self._delayedreaction[drind,numpy.arange(len(drind),dtype=int)]+=Wrow
        else:
            raise ValueError('delay array should be 1 or 2 dimensional')

#    def connect_random(self,P,Q,p,weight=1.,delay=0.,fixed=False, seed=None):
#        Connection.connect_random(self, P, Q, p, weight=weight, fixed=fixed, seed=seed)
#        if isNumberType(delay):
#            self.delays = delay
#        elif callable(delay):
#            n = len(P)
#            m = len(Q)
#            if delay.func_code.co_argcount==0: # TODO: should work with partial objects
#                if self.delayvec.ndim==1:
#                    for i in xrange(n):
#                        self.delays[i] = [delay() for _ in xrange(m)]
#                else:
#                    if isinstance(self.W,SparseConnectionMatrix) and isinstance(self.delayvec, SparseConnectionMatrix):
#                        for i in xrange(n):
#                            self.delayvec.rows[i] = self.W.rows[i]
#                            self.delayvec.data[i] = [delay() for _ in xrange(len(self.delayvec.rows[i]))]
#                    elif isinstance(self.delayvec, DenseConnectionMatrix):
#                        for i in xrange(n):
#                            J = where(self.W[i]!=0.)[0]
#                            self.delayvec[i,J] = [delay() for _ in xrange(len(J))]
#                    else:
#                        raise TypeError('connect_random can only be used with sparse or dense connection matrix types')
#            elif delay.func_code.co_argcount==2 or delay.func_code.co_argcount==1:
#                ac = delay.func_code.co_argcount
#                if self.delayvec.ndim==1:
#                    if ac==2:
#                        raise TypeError('With one-dimensional delay structures, delay function must take one or no arguments')
#                    self.delays = [delay(i) for i in xrange(n)]
#                else:
#                    if isinstance(self.W,SparseConnectionMatrix) and isinstance(self.delayvec, SparseConnectionMatrix):
#                        for i in xrange(n):
#                            J = self.delayvec.rows[i] = self.W.rows[i]
#                            if ac==1:
#                                d = [delay(i)] * len(J)
#                            else:
#                                d = delay(i, J)
#                            self.delayvec.data[i] = d
#                    elif isinstance(self.delayvec, DenseConnectionMatrix):
#                        for i in xrange(n):
#                            J = where(self.W[i]!=0.)[0]
#                            if ac==1:
#                                d = [delay(i)] * len(J)
#                            else:
#                                d = delay(i, J)
#                            self.delayvec[i,J] = d
#                    else:
#                        raise TypeError('connect_random can only be used with sparse or dense connection matrix types')
#            else:
#                raise TypeError,"Bad number of arguments in delay function (should be 0, 1 or 2)"
#        else:
#            raise TypeError('delays should be number, array or function')
    
    def set_delays(self, val):
        self.delayvec[:]=val

    delays = property(fget=lambda self:self.delayvec, fset=set_delays)

#    def compress(self):
#        if not self.iscompressed:
#            Connection.compress(self)
#            if self.delayvec.ndim==2:
#                self.delayvec.freeze()    


class PresynapticDelayConnection(Connection):
    def __init__(self, source, target, state=0, modulation=None,
                 structure='sparse', max_delay=5*msecond, delay_dimension=1, **kwds):
        Connection.__init__(self, source, target, state=state, modulation=modulation, structure=structure, **kwds)
        self._max_delay = int(max_delay/target.clock.dt)+1
        # Each row of the following array stores the cumulative effect of spikes at some
        # particular time, defined by a circular indexing scheme. The _cur_delay_ind attribute
        # stores the row corresponding to the current time, so that _cur_delay_ind+1 corresponds
        # to that time + target.clock.dt, and so on. When _cur_delay_ind reaches _max_delay it
        # resets to zero.
        self._delayedreaction = numpy.zeros((self._max_delay, len(source)),dtype=bool)
        # vector of delay times, can be changed during a run
        if delay_dimension==1:
            self.delayvec = numpy.zeros(len(source))
#        elif delay_dimension==2:
#            if isinstance(structure,str):
#                structure = {'sparse':SparseConnectionMatrix,
#                    'dense':DenseConnectionMatrix,
#                    'computed':ComputedConnectionMatrix
#                    }[structure]
#            self.delayvec=structure((len(source),len(target)),**kwds)
##            self.delayvec = numpy.zeros((len(source), len(target)))
        else:
            raise ValueError('delay_dimension must be 1')
        self._cur_delay_ind = 0
        # this network operation is added to the Network object via the contained_objects
        # protocol (see the line after the function definition). The standard Connection.propagate
        # function propagates spikes to _delayedreaction rather than the target, and this
        # function which is called after the usual propagations propagates that data from
        # _delayedreaction to the target. It only needs to be called each target.clock update.
        @network_operation(clock=target.clock, when='after_connections')
        def delayed_propagate():
            # propagate from _delayedreaction -> target group multiplying by W
            Connection.propagate(self, where(self._delayedreaction[self._cur_delay_ind])[0])
            # reset the current row of _delayedreaction
            self._delayedreaction[self._cur_delay_ind, :] = False
            # increase the index for the circular indexing scheme
            self._cur_delay_ind = (self._cur_delay_ind + 1) % self._max_delay
        self.contained_objects = [delayed_propagate]
        # this is just used to convert delayvec's which are in ms to integers, precalculating it makes it faster
        self._invtargetdt = 1/self.target.clock._dt
        self._useaccel = get_global_preference('useweave')
        self._cpp_compiler = get_global_preference('weavecompiler')
    def propagate(self, spikes):
        # TODO: modulation not yet supported
        if self.delayvec.ndim==1:
            if not isinstance(spikes, numpy.ndarray):
                spikes = array(spikes, dtype=int)
            dvec = self.delayvec[spikes]
            drind = (self._cur_delay_ind+numpy.array(self._invtargetdt*dvec,dtype=int))%self._max_delay
            self._delayedreaction[drind, spikes] = True
#        elif self.delayvec.ndim==2:
#            for i in spikes:
#                # TODO: fix this temporary hack until we sort out generalised connection matrices better
#                dvecrow = self.delayvec[i,:]
#                # handle the case where W is a sparse matrix
#                if hasattr(dvecrow,'toarray'):
#                    dvecrow = dvecrow.toarray().squeeze()
#                drind = (self._cur_delay_ind+numpy.array(self._invtargetdt*dvecrow,dtype=int))%self._max_delay
#                # TODO: fix this temporary hack until we sort out generalised connection matrices better
#                Wrow = self.W[i,:]
#                # handle the case where W is a sparse matrix
#                if hasattr(Wrow,'toarray'):
#                    Wrow = Wrow.toarray().squeeze()
#                self._delayedreaction[drind,numpy.arange(len(drind),dtype=int)]+=Wrow
        else:
            raise ValueError('delay array should be 1 dimensional')
    
    def set_delays(self, val):
        self.delayvec[:]=val

    delays = property(fget=lambda self:self.delayvec, fset=set_delays)

if __name__=='__main__':
    
    dimensions = 1
    dc = PresynapticDelayConnection
    
    if dimensions==1:
    
        maxdelay = 5*ms
        N = 100
        
        inp = SpikeGeneratorGroup(N, [(i, 0*ms) for i in range(N)])
        outp = NeuronGroup(N, model='V:1', threshold=0.5, reset=0.0)
        
        C = dc(inp, outp, delay_dimension=1)
        C.connect_one_to_one(inp, outp)
        C.delays = [(maxdelay*i)/(N-1) for i in range(N)]
        
        M_outp = SpikeMonitor(outp)
        
        run(20*ms)
        
        raster_plot()
        show()
    
    else:
        
        maxdelay = 5*ms
        N = 100
        M = 100
        
        inp = SpikeGeneratorGroup(N, [(i, 0*ms) for i in range(N)])
        outp = NeuronGroup(M, model='V:1', threshold=0.5, reset=0.0)
        
        C = dc(inp, outp, structure='dense', delay_dimension=2)
        C.connect_full(inp, outp)
        for j in range(M):
            C.delays[:, j] = [(maxdelay*i*j)/((N-1)*(M-1)) for i in range(N)]
        C.delays[0,1] = 3*ms
        C.delays[0,2] = 2*ms
            
#        print C.delayvec[4,:]/ms
        
        M_outp = SpikeMonitor(outp)
        
        run(20*ms)
        
#        print M_outp.spikes
        
        raster_plot()
        show()
