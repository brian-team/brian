from brian import *
from scipy import signal, weave, random
from ..bufferable import Bufferable

__all__ = ['Filterbank',
           'RestructureFilterbank',
           'FunctionFilterbank',
           'SumFilterbank',
           'DoNothingFilterbank',
           ]

class Filterbank(Bufferable):
    '''
    Generalised filterbank object
    
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
    
    Also defines arithmetical operations for +, -, *, / where the other
    operand can be a filterbank or scalar. TODO: add more arithmetical ops?
    e.g. could have **, __abs__.
    
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
    
    source = NotImplemented
    
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

    def buffer_init(self):
        Bufferable.buffer_init(self)
        if isinstance(self.source, Bufferable):
            self.source.buffer_init()
        else:
            for s in self.source:
                s.buffer_init()

    def buffer_apply(self, input):
        raise NotImplementedError

    def buffer_fetch_next(self, samples):
        start = self.cached_buffer_end
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
        
    TODO: is this documentation clear enough?
    '''
    def __init__(self, source, numrepeat=1, type='serial', numtile=1,
                 indexmapping=None):
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
        self.indexmapping = indexmapping
        self.nchannels = len(indexmapping)
        self.samplerate = source[0].samplerate
        for s in source:
            if int(s.samplerate)!=int(self.samplerate):
                raise ValueError('All sources must have the same samplerate.')
        self.source = source            

    def buffer_fetch_next(self, samples):
        start = self.cached_buffer_end
        end = start+samples
        inputs = tuple(s.buffer_fetch(start, end) for s in self.source)
        input = hstack(inputs)
        input = input[:, self.indexmapping]
        return input

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
    the number of channels), you will have to derive a class from the
    base class ``Filterbank``).
    '''
    def __init__(self, source, func):
        if isinstance(source, Bufferable):
            source = (source,)
        Filterbank.__init__(self, source)
        self.func = func

    def buffer_fetch_next(self, samples):
        start = self.cached_buffer_end
        end = start+samples
        inputs = tuple(s.buffer_fetch(start, end) for s in self.source)
        return self.func(*inputs)


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
    '''
    def buffer_apply(self, input):
        return input
