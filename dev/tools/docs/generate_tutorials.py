"""Generate re_structured_text data for the tutorials

Reads Brian/Examples/Tutorials/tutorial*.py and generates the file
briantutorials.txt as restructuredtext for inclusion into the main
documentation file brian.txt. This file assumes the tutorial files
are organised as follows:

'''
Text explaining the next bit of code
'''

code

'''
More text
'''

more code

etc.

The text fragments are just inserted directly into the file, and
so we should use the following sectioning standard:

Tutorial title
**************

Tutorial section
~~~~~~~~~~~~~~~~

The code fragments are reformatted according to the restructedtext
format, which is simply that you write

::

    Code
    Lines
    Here

(i.e. :: followed by a tab indented block of code).

"""

import glob # generates directory listing
import os

os.chdir('../../../docs_sphinx') # work from docs_sphinx/ directory until we make the big change

# this takes a file name and returns a list of strings representing the lines
# to be output to the restructuredtext file, and the title of the tutorial
def generate_tutorial(fname):
    title=fname
    f=open(fname)
    lines=f.readlines()
    foundtitle=False
    # we use mode='code' and mode='text' to switch between processing text to be
    # inserted directly (which corresponds to the ''' comment ''' text in the
    # tutorial, and code which needs to be reformatted
    mode='code'
    tut=[] # a list of pairs (mode,lines)
    curlines=[] # the current list of lines to be added to the next (mode,lines) pair
    for l in lines:
        if l[0:3]=="'''" or l[0:3]=='"""': # a mode switch
            if len(curlines): # if there are lines to write then write them
                tut.append((mode, curlines))
            curlines=[]
            if mode=='code':
                mode='text'
            elif mode=='text':
                mode='code'
        else:
            if not foundtitle:
                title=l # Assumption: first line is the title
                foundtitle=True
            curlines.append(l)
    # make sure we've got the last block
    if len(curlines):
        tut.append((mode, curlines))
    tutlines=[] # the output of the function, lines of restructuredtext
    for mode, lines in tut:
        if mode=='code':
            # reformat by adding a paragraph with just :: before it
            tutlines.append('\n')
#            tutlines.append('.. class:: tutorialcode\n')
#            tutlines.append('..\n\n')
            tutlines.append('  ::\n\n')
            for l in lines:
                if len(l.strip()):
                    #tutlines.append('    :literal:`'+l[:-1]+'`\n') # add a tab of 4 spaces before each code line and mark it literal
                    tutlines.append('    '+l[:-1]+'\n') # add a tab of 4 spaces before each code line and mark it literal
                else:
                    tutlines.append('    \n') # add a tab of 4 spaces before each code line
            tutlines.append('\n')
        if mode=='text':
            tutlines.extend(lines)
    return (tutlines, title)

# these are all the tutorial files, when there are more than 9 of them we'll need to add
# an ordering here
tutorial_files=[]
tutorial_series=[]
for roots, dirs, files in os.walk('../tutorials'):
    newdirs=[_ for _ in dirs if '.svn' not in _]
    if '.svn' not in roots and len(newdirs):
        tutorial_series=newdirs
    if 'tutorial' in roots and '.svn' not in roots:
        for f in files:
            if 'tutorial' in f and '.py' in f:
                relpath=os.path.normpath(os.path.join(roots, f))
                tutrelpath=relpath
                tutrelpath=tutrelpath.replace('../tutorials/', '')
                tutrelpath=tutrelpath.replace('..\\tutorials\\', '')
                if '/' in tutrelpath or '\\' in tutrelpath:
                    tutorial_files.append(relpath)

for seriesname in tutorial_series:
    tutdesc=open('../tutorials/'+seriesname+'/introduction.txt', 'r').read()
    seriesfile=open(seriesname+'.txt', 'w')
    seriesfile.write('.. currentmodule:: brian\n\n')
    seriesfile.write('.. _'+seriesname+':\n\n')
    seriesfile.write(tutdesc)
    seriesfile.write('\n\n**Tutorial contents**\n\n')
    seriesfile.write('.. toctree::\n')
    seriesfile.write('   :maxdepth: 2\n\n')
    for fname, i in zip(tutorial_files, range(1, 1+len(tutorial_files))):
        if seriesname in fname:
            tutlines, tutorial_title=generate_tutorial(fname)
            # assumption: case is significant and tutorials are in a subfolder of tutorials with name including tutorials only once
            # e.g. tutorials/tutorial1-basic-concepts/tutorial1a-basic-concepts.py
            head, tail=os.path.split(fname)
            tutorial_name=tail[:tail.index('.py')]
            spaces=5-len(str(i))
            curtut_txt=open(tutorial_name+'.txt', 'w')
            curtut_txt.write('.. currentmodule:: brian\n\n')
            for line in tutlines:
                curtut_txt.write(line)
            del curtut_txt
            seriesfile.write('   '+tutorial_name+'\n')
    seriesfile.close()
