"""Generate re_structured_text interface definition documentation

For each of the interface definition and test functions defined
below in interface_definitions, it generates a corresponding file
interface-modulename.txt containing the assured interface.
The idea is that each module
should define a function _define_and_test_interface which is used by
the test suite Tests/testinterface.py to check that the
interface behaves as expected. These functions should also
contain a textual description of the interface in their docstrings,
and this file reads those docstrings and generates restructuredtext
output from them.
"""

import brian
import os
from brian.utils.documentation import *
from inspect import *

os.chdir('../../../docs_sphinx') # work from docs_sphinx/ directory until we make the big change

interface_definitions = \
    [\
     brian.clock._define_and_test_interface,
     brian.magic._define_and_test_interface,
     brian.units._define_and_test_interface,
     brian.quantityarray._define_and_test_interface,
     brian.directcontrol._define_and_test_interface,
     brian.unitsafefunctions._define_and_test_interface,
     brian.reset._define_and_test_interface,
     brian.threshold._define_and_test_interface,
     brian.monitor._define_and_test_interface,
     brian.connection._define_and_test_interface
    ]

main_interface_page_text = open('interface.txt','r').read()
main_interface_page_text = main_interface_page_text[:main_interface_page_text.find('.. ASSURED_INTERFACE_TOCTREE')]
main_interface_page_text += '.. ASSURED_INTERFACE_TOCTREE\n\n'
main_interface_page_text += '.. toctree::\n'
main_interface_page_text += '   :maxdepth: 2\n\n'

for interface in interface_definitions:
    title = getmodule(interface).__name__
    title = title.replace('brian.','')
    
    main_interface_page_text += '   interface-' + title + '\n'
    
    fname = 'interface-' + title + '.txt' 
    linesout = flattened_docstring(interface.__doc__,split=True)
    outfile = open(fname,'w')
    outfile.write('.. currentmodule:: brian\n\n')
    outfile.write(title+'\n')
    outfile.write('*'*len(title)+'\n\n')
    for l in linesout:
        # little fix for the problem that name_ has a meaning in ReST so we escape the _ in that case
        l = l + '\n'
        l = l.replace('_ ','\_ ')
        l = l.replace('_\n','\_\n')
        outfile.write(l)
    outfile.close()

open('interface.txt','w').write(main_interface_page_text)

#
#all_interfaces = open('all-interfaces.txt','w')
#for interface in interface_definitions:
#    title = getmodule(interface).__name__
#    title = title.replace('brian.','')
#    fname = 'interface-' + title + '.txt' 
#    
#    all_interfaces.write(title+'\n'+'*'*len(title)+'\n\n')
#    all_interfaces.write('.. include:: '+fname+'\n\n')
#    
#    linesout = flattened_docstring(interface.__doc__,split=True)
##    linesout.append('')
##    linesout.append('::')
##    linesout.append('')
##    codelines = getsource(interface).replace(interface.__doc__,'')
##    codelines = codelines.split('\n')
##    codelines = codelines[2:]
##    linesout.extend(codelines)
#    outfile = open(fname,'w')
#    for l in linesout:
#        # little fix for the problem that name_ has a meaning in ReST so we escape the _ in that case
#        l = l + '\n'
#        l = l.replace('_ ','\_ ')
#        l = l.replace('_\n','\_\n')
#        outfile.write(l)