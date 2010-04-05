from brian import *
from nose.tools import *

def test():
    '''
    Equations module.
    '''
    reinit_default_clock()

    tau=2*ms

    eqs=Equations('''
    x=v**2 : volt**2
    y=x # a comment
    dv/dt=(y-v)/tau : volt
    z : amp''')
    
    # Parsing and building
    assert eqs._eq_names==['x','y'] # An alias is also a static equation
    assert eqs._diffeq_names==['v','z'] # A parameter is a differential equation dz/dt=0
    assert eqs._diffeq_names_nonzero==['v']
    assert eqs._alias=={'y':'x'}
    assert eqs._units=={'t':second,'x':volt**2,'z':amp,'y':volt**2, 'v': volt}
    assert eqs._string=={'y': 'x', 'x': 'v**2', 'z': '0*amp/second', 'v': '(y-v)/tau'}
    assert eqs._namespace['v']['tau']==2*ms
    assert 'tau' not in eqs._namespace['x']
    
    # Name substitutions
    assert Equations('dx/dt=-x/(2*ms):1',x='y')._diffeq_names==['y']
    
    # Explicit namespace
    eqs2=Equations('dx/dt=-x/tau:volt',tau=1*ms)
    assert eqs2._namespace['x']['tau']==1*ms
    assert eqs2._namespace['x']=={'tau':1*ms,'volt':volt}
    
    # Find membrane potential
    assert eqs.get_Vm()=='v'
    assert Equations('v=x**2 : 1').get_Vm()==None # must be a differential equation
    assert Equations('dx/dt=1/(2*ms) : 1').get_Vm()==None
    assert Equations('dvm/dt=1/(2*ms) : 1').get_Vm()=='vm'
    assert Equations('''dvm/dt=1/(2*ms) : 1
    dv/dt=1/(2*ms) : 1
    ''').get_Vm()==None # ambiguous
    
    # Unit checking
    eqs=Equations('dv/dt=-v : volt')
    assert_raises(DimensionMismatchError, eqs.prepare)

if __name__=='__main__':
    test()
