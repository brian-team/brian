# ----------------------------------------------------------------------------------
# Copyright ENS, INRIA, CNRS
# Contributors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
# 
# Brian is a computer program whose purpose is to simulate models
# of biological neural networks.
# 
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# ----------------------------------------------------------------------------------
# 
"""
Various utilities for documentation
"""

def indent_string(s, numtabs=1, spacespertab=4, split=False):
    """
    Indents a given string or list of lines
    
    split=True returns the output as a list of lines
    """
    indent = ' ' * (numtabs * spacespertab)
    if isinstance(s, str):
        indentedstring = indent + s.replace('\n', '\n' + indent)
    else:
        indentedstring = ''
        for l in s:
            indentedstring += indent + l + '\n'
    indentedstring = indentedstring.rstrip() + '\n'
    if split: return indentedstring.split('\n')
    return indentedstring

def flattened_docstring(docstr, numtabs=0, spacespertab=4, split=False):
    """
    Returns a docstring with the indentation removed according to the Python standard
    
    split=True returns the output as a list of lines
    
    Changing numtabs adds a custom indentation afterwards
    """
    if isinstance(docstr, str):
        lines = docstr.split('\n')
    else:
        lines = docstr
    if len(lines) < 2: # nothing to do
        return docstr
    flattenedstring = ''
    # Interpret multiline strings according to the Python docstring standard 
    indentlevel = min(# the smallest number of whitespace characters in the lines of the description
                      map(# the number of whitespaces at the beginning of each string in the lines of the description
                          lambda l:len(l) - len(l.lstrip()), # the number of whitespaces at the beginning of the string
                          filter(# only those lines with some text on are counted
                                 lambda l:len(l.strip())
                                 , lines[1:] # ignore the first line
                                 )))
    if lines[0].strip(): # treat first line differently (probably has nothing on it)
        flattenedstring += lines[0] + '\n'
    for l in lines[1:-1]:
        flattenedstring += l[indentlevel:] + '\n'
    if lines[-1].strip(): # treat last line differently (probably has nothing on it)
        flattenedstring += lines[-1][indentlevel:] + '\n'
    return indent_string(flattenedstring, numtabs=numtabs, spacespertab=spacespertab, split=split)

def rest_section(name, level='-', split=False):
    """
    Returns a restructuredtext section heading
    
    Looks like:
    
        name
        ----
        
    """
    s = name + '\n' + level * len(name) + '\n\n'
    if split:
        return s.split('\n')
    return s

if __name__ == '__main__':
    print rest_section('Test'),
    print flattened_docstring(flattened_docstring.__doc__),
