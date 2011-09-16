from brian import *
from scipy import signal, weave, random
from ..bufferable import Bufferable
from operator import isSequenceType
from __builtin__ import all

__all__ = ['Filterbank',
           'RestructureFilterbank',
                'Repeat', 'Tile', 'Join', 'Interleave',
           'FunctionFilterbank',
           'SumFilterbank',
           'DoNothingFilterbank',
           'ControlFilterbank',
           'CombinedFilterbank',
           ]

class Filterbank(Bufferable):
    '''
    Generalised filterbank object
    
    **Documentation common to all filterbanks**
    
    Filterbanks all share a few basic attributes:
    
    .. autoattribute:: source
    
    .. attribute:: nchannels
    
        The number of channels.
        
    .. attribute:: samplerate
    
        The sample rate.
        
    .. autoattribute:: duration

    To process the output of a filterbank, the following method can be used:
    
    .. automethod:: process
    
    Alternatively, the buffer interface can be used, which is described in
    more detail below.
    
    Filterbank also defines arithmetical operations for +, -, ``*``, / where the other
    operand can be a filterbank or scalar.
    
    **Details on the class**
    
    This class is a base class not designed to be instantiated. A Filterbank
    object should define the interface of :class:`Bufferable`, as well as
    defining a ``source`` attribute. This is normally a :class:`Bufferable`
    object, but could be an iterable of sources (for example, for filterbanks
    that mix or add multiple inputs).
    
    The ``buffer_fetch_next(samples)`` method has a default implementation
    that fetches the next input, and calls the ``buffer_apply(input)``
    method on it, which can be overridden by a derived class. This is typically
    the easiest way to implement a new filterbank. Filterbanks with multiple
    sources will need to override this default implementation.
    
    There is a default ``__init__`` method that can be called by a derived class
    that sets the ``source``, ``nchannels`` and ``samplerate`` from that of the
    ``source`` object. For multiple sources, the default implementation will
    check that each source has the same number of channels and samplerate and
    will raise an error if not.
    
    There is a default ``buffer_init()`` method that calls ``buffer_init()`` on
    the ``source`` (or list of sources).
        
    **Example of deriving a class**
    
    The following class takes N input channels and sums them to a single output
    channel::
    
        class AccumulateFilterbank(Filterbank):
            def __init__(self, source):
                Filterbank.__init__(self, source)
                self.nchannels = 1
            def buffer_apply(self, input):
                return reshape(sum(input, axis=1), (input.shape[0], 1))
            
    Note that the default ``Filterbank.__init__`` will set the number of
    channels equal to the number of source channels, but we want to change it
    to have a single output channel. We use the ``buffer_apply`` method which
    automatically handles the efficient cacheing of the buffer for us. The
    method receives the array ``input`` which has shape ``(bufsize, nchannels)``
    and sums over the channels (``axis=1``). It's important to reshape the
    output so that it has shape ``(bufsize, outputnchannels)`` so that it can
    be used as the input to subsequent filterbanks.
    '''
       
    def __init__(self, source):
        if isinstance(source, Bufferable):
            self.source = source
            self.nchannels = source.nchannels
            self.samplerate = source.samplerate
        else:
            self.nchannels = source[0].nchannels
            self.samplerate = source[0].samplerate
            for s in source:
                if s.nchannels!=self.nchannels:
                    raise ValueError('All sources must have the same number of channels.')
                if int(s.samplerate)!=int(self.samplerate):
                    raise ValueError('All sources must have the same samplerate.')
            self.source = source

    def change_source(self, source):
        if not hasattr(self, '_source') or self._source is None:
            self._source = source
            return
        if isinstance(source, tuple):
            for s in source:
                if int(s.samplerate)!=int(self.samplerate):
                    raise ValueError('source samplerate is wrong.')
            for news, olds in zip(source, self._source):
                if news.nchannels!=olds.nchannels:
                    raise ValueError('New sources have different numbers of channels to old sources.')
            self._source = source
            return
        if source.nchannels==self.nchannels:
            self._source = source
            return
        if source.nchannels==1:
            self._source = Repeat(source, self.nchannels)
        else:
            raise ValueError('New source must have the same number of channels as old source.')

    source = property(fget=lambda self:self._source,
                      fset=lambda self, source:self.change_source(source),
                      doc='''
        The source of the filterbank, a :class:`Bufferable` object, e.g. another
        :class:`Filterbank` or a :class:`Sound`. It can also be a tuple of 
        sources. Can be changed after the object
        is created, although note that for some filterbanks this may cause
        problems if they do make assumptions about the input based on the first
        source object they were passed. If this is causing problems, you can
        insert a dummy filterbank (:class:`DoNothingFilterbank`) which is
        guaranteed to work if you change the source.
        ''')
    
    def get_duration(self):
        if hasattr(self, '_duration'):
            return self._duration
        else:
            source = self.source
            if isinstance(source, Bufferable):
                source = [source]
            try:
                durations = [s.duration for s in source]
                duration = max(durations)
                return duration
            except KeyError:
                raise KeyError('Cannot compute duration from sources.')

    def set_duration(self, duration):
        self._duration = duration
    
    duration = property(fget=get_duration, fset=set_duration, doc='''
        The duration of the filterbank. If it is not specified by the user, it
        is computed by finding the maximum of its source durations. If these are
        not specified a :class:`KeyError` will be raised (for example, using
        :class:`OnlineSound` as a source).
        ''')

    def process(self, func=None, duration=None, buffersize=32):
        '''
        Returns the output of the filterbank for the given duration.
        
        ``func``
            If a function is specified, it should be a function of one or two
            arguments that will be called on each filtered buffered segment
            (of shape ``(buffersize, nchannels)`` in order. If the function has
            one argument, the argument should be buffered segment. If it has
            two arguments, the second argument is the value returned by the
            previous application of the function (or 0 for the first
            application). In this case, the method will return the final
            value returned by the function. See example below.
        ``duration=None``
            The length of time (in seconds) or number of samples to process.
            If no ``func`` is specified, the method will return an array of shape
            ``(duration, nchannels)`` with the filtered outputs. Note that in
            many cases, this will be too large to fit in memory, in which you
            will want to process the filtered outputs online, by providing
            a function ``func`` (see example below). If no duration is specified,
            the maximum duration of the inputs to the filterbank will be used,
            or an error raised if they do not have durations (e.g. in the case
            of :class:`OnlineSound`).
        ``buffersize=32``
            The size of the buffered segments to fetch, as a length of time or
            number of samples. 32 samples typically gives reasonably good
            performance.
            
        For example, to compute the RMS of each channel in a filterbank, you
        would do::
        
            def sum_of_squares(input, running_sum_of_squares):
                return running_sum_of_squares+sum(input**2, axis=0)
            rms = sqrt(fb.process(sum_of_squares)/nsamples)
        '''
        if duration is None:
            duration = self.duration
        if not isinstance(duration, int):
            duration = int(duration*self.samplerate)
        if not isinstance(buffersize, int):
            buffersize = int(buffersize*self.samplerate)
        self.buffer_init()
        endpoints = hstack((arange(0, duration, buffersize), duration))
        zendpoints = zip(endpoints[:-1], endpoints[1:])
        #sizes = diff(endpoints)
        if func is None:
            return vstack(tuple(self.buffer_fetch(start, end) for start, end in zendpoints))
        else:
            if func.func_code.co_argcount==1:
                for start, end in zendpoints:
                    func(self.buffer_fetch(start, end))
            else:
                runningval = 0
                for start, end in zendpoints:
                    runningval = func(self.buffer_fetch(start, end), runningval)
                return runningval

    def buffer_init(self):
        Bufferable.buffer_init(self)
        if isinstance(self.source, Bufferable):
            self.source.buffer_init()
        else:
            for s in self.source:
                s.buffer_init()
        self.next_sample = 0

    def buffer_apply(self, input):
        raise NotImplementedError

    def buffer_fetch_next(self, samples):
        start = self.next_sample
        self.next_sample += samples
        end = start+samples
        input = self.source.buffer_fetch(start, end)
        return self.buffer_apply(input)
    
    def __add__ (self, other):
        if isinstance(other, Bufferable):
            return SumFilterbank((self, other))
        else:
            func = lambda x: other+x
            return FunctionFilterbank(self, func)
    __radd__ = __add__
        
    def __sub__ (self, other):
        if isinstance(other, Bufferable):
            return SumFilterbank((self, other), (1, -1))
        else:
            func = lambda x: x-other
            return FunctionFilterbank(self, func)
        
    def __rsub__ (self, other):
        # Note that __rsub__ should return other-self
        if isinstance(other, Bufferable):
            return SumFilterbank((self, other), (-1, 1))
        else:
            func = lambda x: other-x
            return FunctionFilterbank(self, func)

    def __mul__(self, other):
        if isinstance(other, Bufferable):
            func = lambda x, y: x*y
            return FunctionFilterbank((self, other), func)
        else:
            func = lambda x: x*other
            return FunctionFilterbank(self, func)        
    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, Bufferable):
            func = lambda x, y: x/y
            return FunctionFilterbank((self, other), func)
        else:
            func = lambda x: x/other
            return FunctionFilterbank(self, func)        

    def __rdiv__(self, other):
        # Note __rdiv__ returns other/self
        if isinstance(other, Bufferable):
            func = lambda x, y: x/y
            return FunctionFilterbank((other, self), func)
        else:
            func = lambda x: other/x
            return FunctionFilterbank(self, func)        


class RestructureFilterbank(Filterbank):
    '''
    Filterbank used to restructure channels, including repeating and interleaving.
    
    **Standard forms of usage:**
    
    Repeat mono source N times::
    
        RestructureFilterbank(source, N)
        
    For a stereo source, N copies of the left channel followed by N copies of
    the right channel::
    
        RestructureFilterbank(source, N)
        
    For a stereo source, N copies of the channels tiled as LRLRLR...LR::
    
        RestructureFilterbank(source, numtile=N)
        
    For two stereo sources AB and CD, join them together in serial to form the
    output channels in order ABCD::
    
        RestructureFilterbank((AB, CD))
        
    For two stereo sources AB and CD, join them together interleaved to form
    the output channels in order ACBD::
    
        RestructureFilterbank((AB, CD), type='interleave')
        
    These arguments can also be combined together, for example to AB and CD
    into output channels AABBCCDDAABBCCDDAABBCCDD::
    
        RestructureFilterbank((AB, CD), 2, 'serial', 3)
        
    The three arguments are the number of repeats before joining, the joining
    type ('serial' or 'interleave') and the number of tilings after joining.
    See below for details.
    
    **Initialise arguments:**
    
    ``source``
        Input source or list of sources.
    ``numrepeat=1``
        Number of times each channel in each of the input sources is repeated
        before mixing the source channels. For example, with repeat=2 an input
        source with channels ``AB`` will be repeated to form ``AABB``
    ``type='serial'``
        The method for joining the source channels, the options are ``'serial'``
        to join the channels in series, or ``'interleave'`` to interleave them.
        In the case of ``'interleave'``, each source must have the same number
        of channels. An example of serial, if the input sources are ``abc``
        and ``def`` the output would be ``abcdef``. For interleave, the output
        would be ``adbecf``.
    ``numtile=1``
        The number of times the joined channels are tiled, so if the joined
        channels are ``ABC`` and ``numtile=3`` the output will be ``ABCABCABC``.
    ``indexmapping=None``
        Instead of specifying the restructuring via ``numrepeat, type, numtile``
        you can directly give the mapping of input indices to output indices.
        So for a single stereo source input, ``indexmapping=[1,0]`` would
        reverse left and right. Similarly, with two mono sources,
        ``indexmapping=[1,0]`` would have channel 0 of the output correspond to
        source 1 and channel 1 of the output corresponding to source 0. This is
        because the indices are counted in order of channels starting from the
        first source and continuing to the last. For example, suppose you had
        two sources, each consisting of a stereo sound, say source 0 was
        ``AB`` and source 1 was ``CD`` then ``indexmapping=[1, 0, 3, 2]`` would
        swap the left and right of each source, but leave the order of the
        sources the same, i.e. the output would be ``BADC``.        
    '''
    def __init__(self, source, numrepeat=1, type='serial', numtile=1,
                 indexmapping=None):
        self._has_been_optimised = False
        self._reinit(source, numrepeat, type, numtile, indexmapping)
    
    def _do_reinit(self):
        self._reinit(*self._original_init_arguments)
        if self._has_been_optimised:
            self._optimisation_target._do_reinit()
    
    def _reinit(self, source, numrepeat, type, numtile, indexmapping):
        self._original_init_arguments = (source, numrepeat, type, numtile, indexmapping)
        if isinstance(source, Bufferable):
            source = (source,)
        if indexmapping is None:
            nchannels = array([s.nchannels for s in source])
            idx = hstack(([0], cumsum(nchannels)))
            I = [arange(start, stop) for start, stop in zip(idx[:-1], idx[1:])]
            I = tuple(repeat(i, numrepeat) for i in I)
            if type=='serial':
                indexmapping = hstack(I)
            elif type=='interleave':
                if len(unique(nchannels))!=1:
                    raise ValueError('For interleaving, all inputs must have an equal number of channels.')
                I0 = len(I[0])
                indexmapping = zeros(I0*len(I), dtype=int)
                for j, i in enumerate(I):
                    indexmapping[j::len(I)] = i
            else:
                raise ValueError('Type must be "serial" or "interleave"')
            indexmapping = tile(indexmapping, numtile)
        if not isinstance(indexmapping, ndarray):
            indexmapping = array(indexmapping, dtype=int)
        # optimisation to reduce multiple RestructureFilterbanks into a single
        # one, by collating the sources and reconstructing the indexmapping
        # from the individual indexmappings
        if all(isinstance(s, RestructureFilterbank) for s in source):
            newsource = ()
            newsourcesizes = ()
            for s in source:
                s._has_been_optimised = True
                s._optimisation_target = self
                newsource += s.source
                inputsourcesize = sum(inpsource.nchannels for inpsource in s.source)
                newsourcesizes += (inputsourcesize,)
            newsourcesizes = array(newsourcesizes)
            newsourceoffsets = hstack((0, cumsum(newsourcesizes)))
            new_indexmapping = zeros_like(indexmapping)
            sourcesizes = array(tuple(s.nchannels for s in source))
            sourceoffsets = hstack((0, cumsum(sourcesizes)))
            # gives the index of the source of each element of indexmapping
            sourceindices = digitize(indexmapping, cumsum(sourcesizes))
            for i in xrange(len(indexmapping)):
                source_index = sourceindices[i]
                s = source[source_index]
                relative_index = indexmapping[i]-sourceoffsets[source_index]
                source_relative_index = s.indexmapping[relative_index]
                new_index = source_relative_index+newsourceoffsets[source_index]
                new_indexmapping[i] = new_index
            source = newsource
            indexmapping = new_indexmapping
                
        self.indexmapping = indexmapping
        self.nchannels = len(indexmapping)
        self.samplerate = source[0].samplerate
        for s in source:
            if int(s.samplerate)!=int(self.samplerate):
                raise ValueError('All sources must have the same samplerate.')
        self._source = source

    def buffer_fetch_next(self, samples):
        start = self.next_sample
        self.next_sample += samples
        end = start+samples
        inputs = tuple(s.buffer_fetch(start, end) for s in self.source)
        input = hstack(inputs)
        input = input[:, self.indexmapping]
        return input

    def change_source(self, source):
        if not hasattr(self, '_source') or self._source is None:
            self._source = source
            return
        oldsource, numrepeat, type, numtile, indexmapping = self._original_init_arguments
        self._original_init_arguments = source, numrepeat, type, numtile, indexmapping
        self._do_reinit()
#        self._reinit(source, numrepeat, type, numtile, indexmapping)
#        if self._has_been_optimised:
#            target = self._optimisation_target
#            target._reinit(*target._original_init_arguments)

class Repeat(RestructureFilterbank):
    '''
    Filterbank that repeats each channel from its input, e.g. with 3 repeats
    channels ABC would map to AAABBBCCC.
    '''
    def __init__(self, source, numrepeat):
        RestructureFilterbank.__init__(self, source, numrepeat)

class Tile(RestructureFilterbank):
    '''
    Filterbank that tiles the channels from its input, e.g. with 3 tiles
    channels ABC would map to ABCABCABC.
    '''
    def __init__(self, source, numtile):
        RestructureFilterbank.__init__(self, source, numtile=numtile)
        
class Join(RestructureFilterbank):
    '''
    Filterbank that joins the channels of its inputs in series, e.g. with two
    input sources with channels AB and CD respectively, the output would have
    channels ABCD. You can initialise with multiple sources separated by
    commas, or by passing a list of sources.
    '''
    def __init__(self, *sources):
        source = []
        for s in sources:
            if isinstance(s, Bufferable):
                source.append(s)
            else:
                source.extend(s)
        RestructureFilterbank.__init__(self, tuple(source), type='serial')
        
class Interleave(RestructureFilterbank):
    '''
    Filterbank that interleaves the channels of its inputs, e.g. with two
    input sources with channels AB and CD respectively, the output would have
    channels ACBD. You can initialise with multiple sources separated by
    commas, or by passing a list of sources.
    '''
    def __init__(self, *sources):
        source = []
        for s in sources:
            if isinstance(s, Bufferable):
                source.append(s)
            else:
                source.extend(s)
        RestructureFilterbank.__init__(self, tuple(source), type='interleave')

class FunctionFilterbank(Filterbank):
    '''
    Filterbank that just applies a given function. The function should take
    as many arguments as there are sources.
    
    For example, to half-wave rectify inputs::
    
        FunctionFilterbank(source, lambda x: clip(x, 0, Inf))
        
    The syntax ``lambda x: clip(x, 0, Inf)`` defines a function object that
    takes a single argument ``x`` and returns ``clip(x, 0, Inf)``. The numpy
    function ``clip(x, low, high)`` returns the values of ``x`` clipped between
    ``low`` and ``high`` (so if ``x<low`` it returns ``low``, if ``x>high`` it
    returns ``high``, otherwise it returns ``x``). The symbol ``Inf`` means
    infinity, i.e. no clipping of positive values.
    
    **Technical details**
    
    Note that functions should operate on arrays, in particular on 2D buffered
    segments, which are arrays of shape ``(bufsize, nchannels)``. Typically,
    most standard functions from numpy will work element-wise.
    
    If you want a filterbank that changes the shape of the input (e.g. changes
    the number of channels), set the ``nchannels`` keyword argument to the
    number of output channels.
    '''
    def __init__(self, source, func, nchannels=None,**params):
        if isinstance(source, Bufferable):
            source = (source,)
        Filterbank.__init__(self, source)
        self.func = func
        if nchannels is not None:
            self.nchannels = nchannels
        self.params = params

    def buffer_fetch_next(self, samples):
        start = self.cached_buffer_end
        end = start+samples
        inputs = tuple(s.buffer_fetch(start, end) for s in self.source)
#        print inputs,self.params
        return self.func(*inputs,**self.params)


class SumFilterbank(FunctionFilterbank):
    '''  
    Sum filterbanks together with given weight vectors.
    
    For example, to take the sum of two filterbanks::
    
        SumFilterbank((fb1, fb2))
        
    To take the difference::
    
        SumFilterbank((fb1, fb2), (1, -1))
    '''
    def __init__(self, source, weights=None):
        if weights is None:
            weights = ones(len(source))
        self.weights = weights
        func = lambda *inputs: sum(input*w for input, w in zip(inputs, weights))
        FunctionFilterbank.__init__(self, source, func)


class DoNothingFilterbank(Filterbank):
    '''
    Filterbank that does nothing to its input.
    
    Useful for removing a set of filters without having to rewrite your code.
    Can also be used for simply writing compound derived classes. For example,
    if you want a compound Filterbank that does AFilterbank and then
    BFilterbank, but you want to encapsulate that into a single class, you
    could do::
    
        class ABFilterbank(DoNothingFilterbank):
            def __init__(self, source):
                a = AFilterbank(source)
                b = BFilterbank(a)
                DoNothingFilterbank.__init__(self, b)
                
    However, a more general way of writing compound filterbanks is to use
    :class:`CombinedFilterbank`.
    '''
    def buffer_apply(self, input):
        return input


class ControlFilterbank(Filterbank):
    '''
    Filterbank that can be used for controlling behaviour at runtime
    
    Typically, this class is used to implement a control path in an auditory
    model, modifying some filterbank parameters based on the output of other
    filterbanks (or the same ones).
    
    The controller has a set of input filterbanks whose output values are used
    to modify a set of output filterbanks. The update is done by a user specified
    function or class which is passed these output values. The controller should
    be inserted as the last bank in a chain.
    
    Initialisation arguments:
    
    ``source``
        The source filterbank, the values from this are used unmodified as the
        output of this filterbank.
    ``inputs``
        Either a single filterbank, or sequence of filterbanks which are used
        as inputs to the ``updater``.
    ``targets``
        The filterbank or sequence of filterbanks that are modified by the
        updater.
    ``updater``
        The function or class which does the updating, see below.
    ``max_interval``
        If specified, ensures that the updater is called at least as often
        as this interval (but it may be called more often). Can be specified
        as a time or a number of samples.
    
    **The updater**
    
    The ``updater`` argument can be either a function or class instance. If it
    is a function, it should have a form like::
    
        # A single input
        def updater(input):
            ...
        
        # Two inputs
        def updater(input1, input2):
            ...
        
        # Arbitrary number of inputs
        def updater(*inputs):
            ...
            
    Each argument ``input`` to the function is a numpy array of shape
    ``(numsamples, numchannels)`` where ``numsamples`` is the number of samples
    just computed, and ``numchannels`` is the number of channels in the
    corresponding filterbank. The function is not restricted in what it can
    do with these inputs.
    
    Functions can be used to implement relatively simple controllers, but for
    more complicated situations you may want to maintain some state variables
    for example, and in this case you can use a class. The object ``updater``
    should be an instance of a class that defines the ``__call__`` method
    (with the same syntax as above for functions). In addition, you can
    define a reinitialisation method ``reinit()`` which will be called when
    the ``buffer_init()`` method is called on the filterbank, although this is
    entirely optional.
    
    **Example**
    
    The following will do a simple form of gain control, where the gain
    parameter will drift exponentially towards target_rms/rms with a given time
    constant::
    
        # This class implements the gain (see Filterbank for details)
        class GainFilterbank(Filterbank):
            def __init__(self, source, gain=1.0):
                Filterbank.__init__(self, source)
                self.gain = gain
            def buffer_apply(self, input):
                return self.gain*input
        
        # This is the class for the updater object
        class GainController(object):
            def __init__(self, target, target_rms, time_constant):
                self.target = target
                self.target_rms = target_rms
                self.time_constant = time_constant
            def reinit(self):
                self.sumsquare = 0
                self.numsamples = 0
            def __call__(self, input):
                T = input.shape[0]/self.target.samplerate
                self.sumsquare += sum(input**2)
                self.numsamples += input.size
                rms = sqrt(self.sumsquare/self.numsamples)
                g = self.target.gain
                g_tgt = self.target_rms/rms
                tau = self.time_constant
                self.target.gain = g_tgt+exp(-T/tau)*(g-g_tgt)
    
    And an example of using this with an input ``source``, a target RMS of 0.2
    and a time constant of 50 ms, updating every 10 ms::
    
        gain_fb = GainFilterbank(source)
        updater = GainController(gain_fb, 0.2, 50*ms)
        control = ControlFilterbank(gain_fb, source, gain_fb, updater, 10*ms)            
    '''
    def __init__(self, source, inputs, targets, updater, max_interval=None):
        Filterbank.__init__(self, source)
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        if not isinstance(targets, (list, tuple)):
            targets = [targets]
        self.inputs = inputs
        self.updater = updater
        if max_interval is not None:
            if not isinstance(max_interval, int):
                max_interval = int(max_interval*source.samplerate)
            for x in inputs+targets:
                x.maximum_buffer_size = max_interval
            self.maximum_buffer_size = max_interval
            
    def buffer_init(self):
        Filterbank.buffer_init(self)
        if hasattr(self.updater, 'reinit'):
            self.updater.reinit()
    
    def buffer_fetch_next(self, samples):
        start = self.next_sample
        self.next_sample += samples
        end = start+samples
        source_input = self.source.buffer_fetch(start, end)
        input_buffers = [x.buffer_fetch(start, end) for x in self.inputs]
        self.updater(*input_buffers)
        return source_input


class CombinedFilterbank(Filterbank):
    '''
    Filterbank that encapsulates a chain of filterbanks internally.
    
    This class should mostly be used by people writing extensions to Brian hears
    rather than by users directly. The purpose is to take an existing chain of
    filterbanks and wrap them up so they appear to the user as a single
    filterbank which can be used exactly as any other filterbank.
    
    In order to do this, derive from this class and in your initialisation
    follow this pattern::
    
        class RectifiedGammatone(CombinedFilterbank):
            def __init__(self, source, cf):
                CombinedFilterbank.__init__(self, source)
                source = self.get_modified_source()
                # At this point, insert your chain of filterbanks acting on
                # the modified source object
                gfb = Gammatone(source, cf)
                rectified = FunctionFilterbank(gfb,
                                lambda input: clip(input, 0, Inf))
                # Finally, set the output filterbank to be the last in your chain
                self.set_output(fb)
    
    This combination of a :class:`Gammatone` and a rectification via a
    :class:`FunctionFilterbank` can now be used as a single filterbank, for
    example::
    
        x = whitenoise(100*ms)
        fb = RectifiedGammatone(x, [1*kHz, 1.5*kHz])
        y = fb.process()

    **Details**
    
    The reason for the ``get_modified_source()`` call is that the source
    attribute of a filterbank can be changed after creation. The modified source
    provides a buffer (in fact, a :class:`DoNothingFilterbank`) so that the
    input to the chain of filters defined by the derived class doesn't need to
    be changed.
    '''
    def __init__(self, source):
        Filterbank.__init__(self, source)

    def get_duration(self):
        if hasattr(self, '_duration'):
            return self._duration
        else:
            return max(Filterbank.get_duration(self), self.output.duration)        

    source = property(fget=lambda self:self._source,
                      fset=lambda self, source:self.change_source(source))
    
    def change_source(self, source):
        Filterbank.change_source(self, source)
        if hasattr(self, '_modified_source'):
            self._modified_source.source = source

    def get_modified_source(self):
        self._modified_source = DoNothingFilterbank(self.source)
        return self._modified_source
    
    def set_output(self, output):
        self.output = output
        self.nchannels = output.nchannels
                    
    def buffer_init(self):
        Filterbank.buffer_init(self)
        self.output.buffer_init()
            
    def buffer_fetch(self, start, end):
        return self.output.buffer_fetch(start, end)
