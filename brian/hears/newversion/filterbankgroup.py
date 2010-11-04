# TODO: update all of this with the new interface/buffering mechanism

from filterbank import Filterbank
from brian import StateUpdater, NeuronGroup

__all__ = ['FilterbankGroup']

class FilterbankGroupStateUpdater(StateUpdater):
    
    def __init__(self):
        pass

    def __call__(self, P):

        if P._x_stilliter is not None:
            try:
                P.input=P._x_iter.next()
            except StopIteration:
                P.input=0
                P._x_stilliter=False
        elif P._x_type==1:
            P.input=P._x.update()   #for online sound
        elif P._x_type==2:
            P.input=P._x  
        

        P.output[:]=P.filterbank.timestep(P.input)

class FilterbankGroup(NeuronGroup):
    '''
    Allows a Filterbank object to be used as a NeuronGroup
    
    Initialised with variables:
    
    ``filterbank``
        The Filterbank object to be used by the group.
    ``x``
        The sound which the Filterbank will act on. If you don't specify
        this then you are in charge of updating the ``inp`` variable of
        the group each time step via a network operation.
    
    The variables of the group are:
    
    ``output``
        The output of the filterbank, multiple names are provided but they all
        do the same thing.
    ``input``
        The input to the filterbank.
    
    Has one additional method:
    
    .. method:: load_sound(x)
    
        Loads the sound 
    '''
    
    def __init__(self, filterbank, x=None):
        self.filterbank=filterbank
        fs=filterbank.samplerate
        eqs='''
        output : 1
        input : 1
        '''

        NeuronGroup.__init__(self, len(filterbank), eqs, clock=Clock(dt=1/fs))
        self.N=len(filterbank)
        self._state_updater=FilterbankGroupStateUpdater()
        fs=float(fs)
        self.load_sound(x)

    def load_sound(self, x):
        self._x=x
        
        if isinstance(x,OnlineSound):
            self._x_iter=None
            self._x_type=1        #type=1 for online sounds
            self._x_stilliter=None
            
        elif x is not None:           #hack to be able to plug a 1d filterbankgroup in another
            if len(self._x)==1:
                self._x_iter=None
                self._x_stilliter=None
                self._x_type=2    #type=2 when the input is the output of a 1d filter chain
            else:   
                self._x_iter=iter(self._x)
                self._x_stilliter=True
                self._x_type=3    #type=3 when input is an array
        else:

            self._x_iter=None
            self._x_stilliter=False

        
    def reinit(self):
        NeuronGroup.reinit(self)
        self.load_sound(self._x)
