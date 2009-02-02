'''
NOTE: you need a .pypirc file to do this, you may need to set the
HOME env to where it is saved. Also note that any spaces in the
filename of HOME will cause it to not work, so use old style 8.3
equivalent name.

Also note that manifest.in may not work right with this?
'''
import os
os.chdir('../../../.') # work from Brian's root
os.system('setup.py register')
os.system('setup.py sdist bdist_wininst upload')
