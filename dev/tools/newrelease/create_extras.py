"""
Generates the file dist/brian-versionnumber-extras.zip from the folders defined in the
setup.py module
"""

import os
import glob

from zipfile import ZipFile

from brian_setup_info import version, extras_folders

pathname = os.path.abspath(os.path.dirname(__file__))
os.chdir(pathname)
os.chdir('../../../.')# work from Brian's root

zipfilename = 'dist/brian-' + version + '-extras.zip'
files = []
for folder in extras_folders:
    files.extend(glob.glob(folder))
zfile = ZipFile(zipfilename, 'w')
for file in files:
    zfile.write(file)
zfile.close()
