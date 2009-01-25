# Short-term plasticity
# See BEP-1
# TO BE TESTED WITH FACILITATION
from network import NetworkOperation
from neurongroup import NeuronGroup
from monitor import SpikeMonitor
from scipy import zeros,exp

__all__=['STP']

class STPGroup(NeuronGroup):
    '''
    Neuron group forwarding spikes with short term plasticity modulation.
    '''
    def __init__(self,N):
        eqs='''
        x : 1
        y : 1
        '''
        NeuronGroup.__init__(self,N,model=eqs)
        
    def update(self):
        pass

class STPUpdater(SpikeMonitor):
    '''
    Event-driven updates of STP variables.
    '''
    def __init__(self,source,P,taud,tauf,U,delay=0):
        SpikeMonitor.__init__(self,source,record=False,delay=delay)
        # P is the group with the STP variables
        N=len(P)
        self.P=P
        self.minvtaud=-1./taud
        self.minvtauf=-1./tauf
        self.U=U
        self.x=P.x
        self.y=P.y
        self.lastt=zeros(N) # last update
        self.clock=P.clock
        
    def propagate(self,spikes):
        interval=self.clock.t-self.lastt[spikes]
        #self.u[spikes]=self.U+(self.u[spikes]-self.U)*exp(interval*self.minvtauf)
        self.y[spikes]*=exp(interval*self.minvtauf)
        tmp=(1-self.U)-self.y[spikes]
        self.x[spikes]=(1+(self.x[spikes]-1)*exp(interval*self.minvtaud))*tmp
        self.y[spikes]+=self.U*tmp
        self.lastt[spikes]=self.clock.t
        self.P.LS.push(spikes)

class STP(NetworkOperation):
    '''
    Short-term synaptic plasticity, following the Tsodyks-Markram model:
    dx/dt=(1-x)/taud  (depression)
    du/dt=(U-u)/tauf  (facilitation)
    spike: x->x*(1-u); u->u+U*(1-u)  (in what order?)
    x is the modulation factor (in 0..1) for the synaptic weight
    
    Rewritten as follows for (minor) optimization:
    dx/dt=(1-x)/taud
    dy/dt=-y/tauf   (y=u-U)
    spike: x->x*(1-U-y); y->y+U*(1-U-y)  (in what order?)
    
    TODO: manage delays correctly
    '''
    def __init__(self,C,taud,tauf,U):
        NetworkOperation.__init__(self,lambda:None)
        N=len(C.source)
        P=STPGroup(N)
        P.x=1
        P.y=0
        self.contained_objects=[STPUpdater(C.source,P,taud,tauf,U)]
        C.source=P
        C._nstate_mod=0 # modulation of synaptic weights
        
    def __call__(self):
        pass
