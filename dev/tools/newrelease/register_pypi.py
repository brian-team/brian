import os
os.chdir('../../../.') # work from Brian's root
os.system('setup.py register')
# NOTE: you need a .pypirc file to do this, you may need to set the HOME env to where it is saved
os.system('setup.py sdist bdist_wininst upload')
