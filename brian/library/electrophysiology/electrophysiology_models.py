'''
Electrophysiology models for Brian.
R. Brette 2008.

Contains:
* Electrode models
* Current clamp and voltage amplifiers
* DCC and SEVC amplifiers (discontinuous)
* AEC (Active Electrode Compensation)
* Acquisition board
'''
from brian.units import amp, check_units, volt, farad
from brian.equations import Equations, unique_id
from operator import isSequenceType
from brian.units import ohm, Mohm
from brian.stdunits import pF, ms, nA, mV, nS
from brian.neurongroup import NeuronGroup
from scipy import zeros, array, optimize, mean, arange, diff, rand, exp, sum, convolve, eye, linalg, sqrt
from brian.clock import Clock
from electrode_compensation import *

__all__ = ['electrode', 'current_clamp', 'voltage_clamp', 'DCC', 'SEVC',
         'AcquisitionBoard', 'AEC', 'VC_AEC']

'''
------------
 Electrodes
------------
'''
# TODO: sharp electrode model (cone)
# No unit checking of Re and Ce because they can be lists
@check_units(v_rec=volt, vm=volt, i_inj=amp, i_cmd=amp)
def electrode(Re, Ce, v_el='v_el', vm='vm', i_inj='i_inj', i_cmd='i_cmd'):
    '''
    An intracellular electrode modeled as an RC circuit,
    or multiple RC circuits in series (if Re, Ce are lists).
    v_el = electrode (=recording) potential
    vm = membrane potential
    i_inj = current entering the membrane
    i_cmd = electrode command current (None = no injection)
    Returns an Equations() object.
    '''
    if isSequenceType(Re):
        if len(Re) != len(Ce) or len(Re) < 2:
            raise TypeError, "Re and Ce must have the same length"
        v_mid, i_mid = [], []
        for i in range(len(Re) - 1):
            v_mid.append('v_mid_' + str(i) + unique_id())
            i_mid.append('i_mid_' + str(i) + unique_id())
        eqs = electrode(Re[0], Ce[0], v_mid[0], vm, i_inj, i_mid[0])
        for i in range(1, len(Re) - 1):
            eqs + electrode(Re[i], Ce[i], v_mid[i], v_mid[i - 1], i_mid[i - 1], i_mid[i])
        eqs += electrode(Re[-1], Ce[-1], v_el, v_mid[-1], i_mid[-1], i_cmd)
        return eqs
    else:
        if Ce > 0 * farad:
            return Equations('''
            dvr/dt = ((vm-vr)/Re+ic)/Ce : mV
            ie = (vr-vm)/Re : nA''', vr=v_el, vm=vm, ic=i_cmd, ie=i_inj, \
            Re=Re, Ce=Ce)
        else: # ideal electrode - pb here
            return Equations('''
            vr = vm+Re*ic : volt
            ie = ic : amp''', vr=v_el, vm=vm, ic=i_cmd, ie=i_inj)

'''
------------
 Amplifiers
------------
'''
def voltage_clamp(vm='vm', v_cmd='v_cmd', i_rec='i_rec',
                  i_inj='i_inj', Re=20 * Mohm, Rs=0 * Mohm, tau_u=1 * ms):
    '''
    Continuous voltage-clamp amplifier + ideal electrode
    (= input capacitance is already neutralized).
    
    vm = membrane potential (or electrode) variable
    v_cmd = command potential
    i_inj = injected current (into the neuron)
    i_rec = recorded current (= -i_inj)
    Re = electrode resistance
    Rs = series resistance compensation
    tau_u = delay of series compensation (for stability)
    '''
    return Equations('''
    Irec=-I : amp
    I=(Vc+U-Vr)/Re : amp
    dU/dt=(Rs*I-U)/tau : volt
    ''', Vr=vm, Vc=v_cmd, I=i_inj, Rs=Rs, tau=tau_u, Irec=i_rec, Re=Re)

# TODO: Re, Ce as lists
def current_clamp(vm='vm', i_inj='i_inj', v_rec='v_rec', i_cmd='i_cmd',
                  Re=80 * Mohm, Ce=4 * pF, bridge=0 * ohm, capa_comp=0 * farad,
                  v_uncomp=None):
    '''
    Continuous current-clamp amplifier + electrode.
    
    vm = membrane potential (or electrode) variable
    i_inj = injected current (into the neuron)
    v_rec = recorded potential
    i_cmd = command current
    bridge = bridge resistance compensation
    capa_comp = capacitance neutralization
    Re = electrode resistance
    Ce = electrode capacitance (input capacitance)
    v_uncomp = uncompensated potential (raw measured potential)
    '''
    if capa_comp != Ce:
        return Equations('''
        Vr=U-R*Ic    : volt
        I=(U-V)/Re : amp
        dU/dt=(Ic-I)/(Ce-CC) : volt
        ''', Vr=v_rec, V=vm, I=i_inj, Ic=i_cmd, R=bridge, Ce=Ce, CC=capa_comp, U=v_uncomp, Re=Re)
    else:
        return Equations('''
        Vr=V+(Re-R)*I    : volt
        I=Ic : amp # not exactly an alias because the units of Ic is unknown
        ''', Vr=v_rec, V=vm, I=i_inj, Ic=i_cmd, R=bridge, Re=Re)


class AcquisitionBoard(NeuronGroup):
    '''
    Digital acquisition board (DSP).
    Use: board=AcquisitionBoard(P=neuron,V='V',I='I',clock)
    where
      P = neuron group
      V = potential variable name (in P)
      I = current variable name (in P)
      clock = acquisition clock
    Recording: vm=board.record
    Injecting: board.command=...
    
    Injects I, records V.
    '''
    def __init__(self, P, V, I, clock=None):
        eqs = Equations('''
        record : units_record
        command : units_command
        ''', units_record=P.unit(V), units_command=P.unit(I))
        NeuronGroup.__init__(self, len(P), model=eqs, clock=clock)
        self._P = P
        self._V = V
        self._I = I

    def update(self):
        self.record = self._P.state(self._V) # Record
        self._P.state(self._I)[:] = self.command # Inject


class DCC(AcquisitionBoard):
    '''
    Discontinuous current-clamp.
    Use: board=DCC(P=neuron,V='V',I='I',frequency=2*kHz)
    where
      P = neuron group
      V = potential variable name (in P)
      I = current variable name (in P)
      frequency = sampling frequency
    Recording: vm=board.record
    Injecting: board.command=I
    '''
    def __init__(self, P, V, I, frequency):
        self.clock = Clock(dt=1. / (3. * frequency))
        AcquisitionBoard.__init__(self, P, V, I, self.clock)
        self._cycle = 0

    def set_frequency(self, frequency):
        '''
        Sets the sampling frequency.
        '''
        self.clock.dt = 1. / (3. * frequency)

    def update(self):
        if self._cycle == 0:
            self.record = self._P.state(self._V) # Record
            self._P.state(self._I)[:] = 3 * self.command # Inject
        else:
            self._P.state(self._I)[:] = 0 #*nA
        self._cycle = (self._cycle + 1) % 3


class SEVC(DCC):
    '''
    Discontinuous voltage-clamp.
    Use: board=SEVC(P=neuron,record='V',command='I',frequency=2*kHz,gain=10*nS)
    where
      P = neuron group
      V = potential variable name (in P)
      I = current variable name (in P)
      frequency = sampling frequency
      gain = feedback gain
      gain2 = control gain (integral controller)
    Recording: i=board.record
    Setting the clamp potential: board.command=-20*mV
    '''
    def __init__(self, P, V, I, frequency, gain=100 * nS, gain2=0 * nS / ms):
        DCC.__init__(self, P, V, I, frequency)
        self._J = zeros(len(P)) # finer control
        self._gain = gain
        self._gain2 = gain2

    def update(self):
        if self._cycle == 0:
            self._J += self.clock._dt * self._gain2 * (self._P.state(self._V) - self.command)
            self.record = self._gain * (self._P.state(self._V) - self.command) + self._J
            self._P.state(self._I)[:] = -3 * self.record # Inject
        else:
            self._P.state(self._I)[:] = 0 #*nA
        self._cycle = (self._cycle + 1) % 3

'''
-------------------------------------
 Active Electrode Compensation (AEC)
 
The technique was presented in the following paper:
High-resolution intracellular recordings using a real-time computational model of the electrode
R. Brette, Z. Piwkowska, C. Monier, M. Rudolph-Lilith, J. Fournier, M. Levy, Y. Fregnac, T. Bal, A. Destexhe
Neuron (2008) 59(3):379-91.

-------------------------------------
'''


class AEC(AcquisitionBoard):
    """
    An acquisition board with AEC.
    Warning: only works with 1 neuron (not N).
    Use: board=AEC(neuron,'V','I',clock)
    where
      P = neuron group
      V = potential variable name (in P)
      I = current variable name (in P)
      clock = acquisition clock
    Recording: vm=board.V
    Injecting: board.command(I)
    """

    def __init__(self, P, V, I, clock=None):
        AcquisitionBoard.__init__(self, P, V, I, clock=clock)
        self._estimation = False
        self._compensation = False
        self.Ke = None
        self._Vrec = []
        self._Irec = []

    def start_injection(self, amp=.5 * nA, DC=0 * nA):
        '''
        Start white noise injection for kernel estimation.
        amp = current amplitude
        DC = additional DC current
        '''
        self._amp, self._DC = amp, DC
        self._estimation = True

    def stop_injection(self):
        '''
        Stop white noise injection.
        '''
        self._estimation = False
        self.command = 0 #*nA

    def estimate(self, ksize=150, ktail=50, dendritic=False):
        '''
        Estimate electrode kernel Ke (after injection)
        
        ksize = kernel size (in bins)
        ktail = tail parameter (in bins), indicates the end of the electrode kernel
        '''
        self._ksize = ksize
        self._ktail = ktail
        # Calculate Ke
        vrec = array(self._Vrec) / mV
        irec = array(self._Irec) / nA
        K = full_kernel(vrec, irec, self._ksize)
        self._K = K * Mohm
        if dendritic:
            self.Ke = electrode_kernel_dendrite(K, ktail) * Mohm
        else:
            self.Ke = electrode_kernel_soma(K, ktail) * Mohm
        self._Vrec = []
        self._Irec = []

    def switch_on(self, Ke=None):
        '''
        Switch compensation on, with kernel Ke.
        (If not given: use last kernel)
        '''
        self._compensation = True
        self._lastI = zeros(self._ktail)
        self._posI = 0

    def switch_off(self):
        '''
        Switch compensation off.
        '''
        self._compensation = False

    def update(self):
        AcquisitionBoard.update(self)
        if self._estimation:
            I = 2. * (rand() - .5) * self._amp + self._DC
            self.command = I
            # Record
            self._Vrec.append(self.record[0])
            self._Irec.append(I)
        if self._compensation:
            # Compensate
            self._lastI[self._posI] = self.command[0]
            self._posI = (self._posI - 1) % self._ktail
            self.record[0] = self.record[0] - sum(self.Ke * self._lastI[range(self._posI, self._ktail) +
                                                        range(0, self._posI)])


class VC_AEC(AEC):
    def __init__(self, P, V, I, gain=50 * nS, gain2=0 * nS / ms, clock=None):
        AEC.__init__(self, P, V, I, clock=clock)
        self._gain = gain
        self._gain2 = gain2
        self._J = zeros(len(P))

    def stop_injection(self):
        '''
        Stop white noise injection.
        '''
        self._estimation = False
        self.record = 0 #*nA

    def update(self):
        V = self._P.state(self._V) # Record
        self._P.state(self._I)[:] = -self.record # Inject
        if self._estimation:
            I = 2. * (rand() - .5) * self._amp + self._DC
            self.record = -I
            # Record
            self._Vrec.append(V[0])
            self._Irec.append(I)
        if self._compensation:
            # Compensate
            self._lastI[self._posI] = -self.record[0]
            self._posI = (self._posI - 1) % self._ktail
            V[0] = V[0] - sum(self.Ke * self._lastI[range(self._posI, self._ktail) + range(0, self._posI)])
            self._J += self.clock._dt * self._gain2 * (self.command - V)
            self.record = -(self._gain * (self.command - V) + self._J)

if __name__ == '__main__':
    from brian import *
    taum = 20 * ms
    gl = 20 * nS
    Cm = taum * gl
    eqs = Equations('dv/dt=(-gl*v+i_inj)/Cm : volt') + electrode(50 * Mohm, 10 * pF, vm='v', i_cmd=.5 * nA)
    neuron = NeuronGroup(1, model=eqs)
    M = StateMonitor(neuron, 'v_el', record=True)
    run(100 * ms)
    plot(M.times / ms, M[0] / mV)
    show()
