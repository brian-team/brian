'''
Performs the same simulation (without any random elements) with different
optimizations (C code generation, etc.) and assures that the results are
always identical.
'''
from brian import *
from brian.tests import repeat_with_global_opts
from scipy.sparse import lil_matrix

def setup_test_cobahh():
    ''' load the matrices once to save some time '''
    global Ce_matrix_dense
    global Ci_matrix_dense
    global Ce_matrix_sparse
    global Ci_matrix_sparse    
    global reference_spikes
    
    # Load the connections
    print 'Loading connection matrices'
    we = 6 * nS # excitatory synaptic weight (voltage)
    wi = 67 * nS # inhibitory synaptic weight    
    Ce_matrix_dense = np.loadtxt('Ce_matrix.txt.gz') * we
    Ci_matrix_dense = np.loadtxt('Ci_matrix.txt.gz') * wi
    Ce_matrix_sparse = lil_matrix(Ce_matrix_dense)
    Ci_matrix_sparse = lil_matrix(Ci_matrix_dense)
    
    print 'Loading reference spikes'
    reference_spikes = np.loadtxt('COBAHH_spikes.txt.gz')    
    print reference_spikes.shape

@repeat_with_global_opts([
                          # no C code or code generation,
                          {'useweave': False, 'usecodegen': False},
                          # # use weave but no code generation 
                          {'useweave': True, 'usecodegen': False}, 
                          # use Python code generation
                          {'useweave': False, 'usecodegen': True,
                           'usecodegenthreshold': True,
                           'usecodegenreset': True,
                           'usecodegenstateupdate': True},
                          # use C code generation
                          {'useweave': True, 'usecodegen': True,
                           'usecodegenthreshold': True,
                           'usecodegenreset': True,
                           'usecodegenweave': True,
                           'usecodegenstateupdate': True}
                          ])
def test_cobahh():
    '''
    Test the COBAHH example with different optimizations (weave, etc.).
    '''
    clear(True)
    reinit_default_clock()
    
    # Parameters
    area = 20000 * umetre ** 2
    Cm = (1 * ufarad * cm ** -2) * area
    gl = (5e-5 * siemens * cm ** -2) * area
    El = -60 * mV
    EK = -90 * mV
    ENa = 50 * mV
    g_na = (100 * msiemens * cm ** -2) * area
    g_kd = (30 * msiemens * cm ** -2) * area
    VT = -63 * mV
    # Time constants
    taue = 5 * ms
    taui = 10 * ms
    # Reversal potentials
    Ee = 0 * mV
    Ei = -80 * mV
    
    # The model
    eqs = Equations('''
    dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-\
        g_na*(m*m*m)*h*(v-ENa)-\
        g_kd*(n*n*n*n)*(v-EK))/Cm : volt 
    dm/dt = alpham*(1-m)-betam*m : 1
    dn/dt = alphan*(1-n)-betan*n : 1
    dh/dt = alphah*(1-h)-betah*h : 1
    dge/dt = -ge*(1./taue) : siemens
    dgi/dt = -gi*(1./taui) : siemens
    alpham = 0.32*(mV**-1)*(13*mV-v+VT)/ \
        (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
    betam = 0.28*(mV**-1)*(v-VT-40*mV)/ \
        (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
    alphah = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
    betah = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
    alphan = 0.032*(mV**-1)*(15*mV-v+VT)/ \
        (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
    betan = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
    ''')
    
    P = NeuronGroup(4000, model=eqs,
        threshold=EmpiricalThreshold(threshold= -20 * mV,
                                     refractory=3 * ms),
        implicit=True, freeze=True)
    Pe = P.subgroup(3200)
    Pi = P.subgroup(800)
    
    print 'Setting up connections' 
    Ce = Connection(Pe, P, 'ge')    
    Ce.connect(Pe, P, Ce_matrix_sparse)
    Ci = Connection(Pi, P, 'gi')
    Ci.connect(Pi, P, Ci_matrix_sparse)
    
    # Initialization (non-random)
    P.v = El + ((arange(len(P), dtype=np.float) / len(P) - 0.5) * 20 - 5) * mV
    P.ge = ((arange(len(P), dtype=np.float) / len(P) - 0.5) * 6 + 4) * 10. * nS
    P.gi = ((arange(len(P), dtype=np.float) / len(P) - 0.5) * 48 + 20) * 10. * nS
    
    mon = SpikeMonitor(P)

    print 'Starting simulation'
    run(1 * second, report='text')
    
    # compare results to the saved spikes
    spikes = array(mon.spikes)

    print 'Comparing spikes to reference spikes'
    if spikes.shape != reference_spikes.shape:
        sys.stderr.write('Spike array shapes do not match: %s vs. %s' %
                         (str(spikes.shape), str(reference_spikes.shape)))
        raise AssertionError()
    elif not (spikes == reference_spikes).all():
        sys.stderr.write('Array content is not identical')
        raise AssertionError()
    else:
        print 'All spikes are equal.'

# Note that code generation is currently not implemented for Synapses
@repeat_with_global_opts([
                          # no C code or code generation,
                          {'useweave': False, 'usecodegen': False},
                          # # use weave but no code generation 
                          {'useweave': True, 'usecodegen': False}, 
                          # use Python code generation
                          {'useweave': False, 'usecodegen': True,
                           'usecodegenthreshold': True,
                           'usecodegenreset': True,
                           'usecodegenstateupdate': True},
                          # use C code generation
                          {'useweave': True, 'usecodegen': True,
                           'usecodegenthreshold': True,
                           'usecodegenreset': True,
                           'usecodegenweave': True,
                           'usecodegenstateupdate': True}
                          ])
def test_cobahh_synapses():
    '''
    Same as test_cobahh, but tests everything using the Synapses class.
    '''
    clear(True)
    reinit_default_clock()
    
    # Parameters
    area = 20000 * umetre ** 2
    Cm = (1 * ufarad * cm ** -2) * area
    gl = (5e-5 * siemens * cm ** -2) * area
    El = -60 * mV
    EK = -90 * mV
    ENa = 50 * mV
    g_na = (100 * msiemens * cm ** -2) * area
    g_kd = (30 * msiemens * cm ** -2) * area
    VT = -63 * mV
    # Time constants
    taue = 5 * ms
    taui = 10 * ms
    # Reversal potentials
    Ee = 0 * mV
    Ei = -80 * mV
    
    # The model
    eqs = Equations('''
    dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-\
        g_na*(m*m*m)*h*(v-ENa)-\
        g_kd*(n*n*n*n)*(v-EK))/Cm : volt 
    dm/dt = alpham*(1-m)-betam*m : 1
    dn/dt = alphan*(1-n)-betan*n : 1
    dh/dt = alphah*(1-h)-betah*h : 1
    dge/dt = -ge*(1./taue) : siemens
    dgi/dt = -gi*(1./taui) : siemens
    alpham = 0.32*(mV**-1)*(13*mV-v+VT)/ \
        (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
    betam = 0.28*(mV**-1)*(v-VT-40*mV)/ \
        (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
    alphah = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
    betah = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
    alphan = 0.032*(mV**-1)*(15*mV-v+VT)/ \
        (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
    betan = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
    ''')
    
    P = NeuronGroup(4000, model=eqs,
        threshold=EmpiricalThreshold(threshold= -20 * mV,
                                     refractory=3 * ms),
        implicit=True, freeze=True)
    Pe = P.subgroup(3200)
    Pi = P.subgroup(800)
    
    print 'Setting up connections'
    Ce = Synapses(Pe, P, model='w:1', pre='ge+=w')
    Ce[:, :] = 'Ce_matrix_dense[i, j] != 0'
    Ce.w[:, :] = 'Ce_matrix_dense[i, j]'
    Ci = Synapses(Pi, P, model='w:1', pre='gi+=w')
    Ci[:, :] = 'Ci_matrix_dense[i, j] != 0'
    Ci.w[:, :] = 'Ci_matrix_dense[i, j]'
    
    # Initialization (non-random)
    P.v = El + ((arange(len(P), dtype=np.float) / len(P) - 0.5) * 20 - 5) * mV
    P.ge = ((arange(len(P), dtype=np.float) / len(P) - 0.5) * 6 + 4) * 10. * nS
    P.gi = ((arange(len(P), dtype=np.float) / len(P) - 0.5) * 48 + 20) * 10. * nS
    
    mon = SpikeMonitor(P)

    print 'Starting simulation'
    run(1 * second, report='text')
    
    # compare results to the saved spikes
    spikes = array(mon.spikes)

    print 'Comparing spikes to reference spikes'
    if spikes.shape != reference_spikes.shape:
        sys.stderr.write('Spike array shapes do not match: %s vs. %s' %
                         (str(spikes.shape), str(reference_spikes.shape)))
        raise AssertionError()
    elif not (spikes == reference_spikes).all():
        sys.stderr.write('Array content is not identical')
        raise AssertionError()
    else:
        print 'All spikes are equal.'


if __name__ == '__main__':
    setup_test_cobahh()
    test_cobahh()
    test_cobahh_synapses()