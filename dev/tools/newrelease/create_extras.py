"""
Generates the file dist/brian-versionnumber-extras.zip from the folders defined in the
setup.py module
"""

import os
import glob
from setup import version, extras_folders
from zipfile import ZipFile

zipfilename='dist/brian-'+version+'-extras.zip'

os.chdir('..')
os.chdir('..')
os.chdir('..') # work from Brian's root

files=[]
for folder in extras_folders:
    files.extend(glob.glob(folder))
zfile=ZipFile(zipfilename, 'w')
for file in files:
    zfile.write(file)
zfile.close()
