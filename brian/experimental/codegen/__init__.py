from codegen import *
from integration_schemes import *
from codegen_python import *
from codegen_c import *
from codegen_gpu import *

if __name__ == '__main__':
    if True:
        from brian import *
        eqs = Equations('''
        dV/dt = -W*V/(10*second) : volt 
        dW/dt = -V**2/(1*second) : volt
        ''')
        scheme = exp_euler_scheme
        print 'Equations'
        print '========='
        print eqs
        print 'Scheme'
        print '======'
        for block_specifier, block_code in scheme:
            print block_specifier
            print block_code
        print 'Python code'
        print '==========='
        print PythonCodeGenerator().generate(eqs, scheme)
        print 'C code'
        print '======'
        print CCodeGenerator().generate(eqs, scheme)
        print 'GPU code'
        print '======'
        print GPUCodeGenerator().generate(eqs, scheme)
