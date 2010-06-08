from brian import *

def testepsp():
    """Tests whether an alpha function EPSP works algebraically.
    
    The expected behaviour of the network below is that it should solve the
    following differential equation:
    
    taum   dV/dt = -V + x
    taupsp dx/dt = -x + y
    taupsp dy/dt = -y
            V(0) = 0 volt
            x(0) = 0 volt
            y(0) = y0 volt
        
    This gives the following analytical solution for V (computed with Mathematica):
    
    V(t) = (E^(-(t/taum) - t/taupsp)*(-(E^(t/taum)*t*taum) + 
       E^(t/taum)*t*taupsp - E^(t/taum)*taum*taupsp + 
       E^(t/taupsp)*taum*taupsp)*y0)/(taum - taupsp)^2
    
    This doesn't have an analytical solution for the maximum value of V, but the
    following numerical value was computed with the analytic formula:
    
            Vmax = 0.136889 mvolt  (accurate to that many sig figs)
    at time    t = 1.69735 ms (accurate to +/- 0.00001ms)
    
    The Brian network consists of two neurons, one governed by the differential
    equations given above, the other fires a single spike at time t=0 and is
    connected to the first
    """
    reinit_default_clock()
    clock = Clock(dt=0.1 * ms)
    expected_vmax = 0.136889 * mvolt
    expected_vmaxtime = 1.69735 * msecond
    desired_vmaxaccuracy = 0.001 * mvolt
    desired_vmaxtimeaccuracy = max(clock.dt, 0.00001 * ms)
    taum = 10 * ms
    taupsp = 0.325 * ms
    y0 = 4.86 * mV
    P = NeuronGroup(N=1, model='''
                  dV/dt = (-V+x)*(1./taum) : volt
                  dx/dt = (-x+y)*(1./taupsp) : volt
                  dy/dt = -y*(1./taupsp) : volt
                  ''',
                  threshold=100 * mV, reset=0 * mV)
    Pinit = SpikeGeneratorGroup(1, [(0, 0 * ms)])
    C = Connection(Pinit, P, 'y')
    C.connect_full(Pinit, P, y0)
    M = StateMonitor(P, 'V', record=0)
    run(10 * ms)
    V = M[0]
    Vmax = 0
    Vi = 0
    for i in range(len(V)):
        if V[i] > Vmax:
            Vmax = V[i]
            Vi = i
    Vmaxtime = M.times[Vi] * second
    Vmax = Vmax * volt
    assert abs(Vmax - expected_vmax) < desired_vmaxaccuracy
    assert abs(Vmaxtime - expected_vmaxtime) < desired_vmaxtimeaccuracy

if __name__ == '__main__':
    testepsp()
