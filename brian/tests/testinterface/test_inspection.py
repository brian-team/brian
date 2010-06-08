from brian import *
from brian.inspection import *

def test():
    '''
    Inspection module
    '''
    reinit_default_clock()
    expr = '''
x_12+=y*12 # comment
pour(water,\
   "on desk")'''
    # Check that identifiers are correctly extracted
    assert (get_identifiers(expr) == ('x_12', 'y', 'pour', 'water'))
    # Check that modified variables are correctly extracted
    assert (modified_variables(expr) == ['x_12', 'pour'])

if __name__ == '__main__':
    test()
