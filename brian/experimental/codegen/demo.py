from brian import *
from brian.experimental.codegen import *
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
