import os
os.chdir('../../../.') # work from Brian's root
os.system('setup.py bdist_wininst')
os.system('setup.py sdist --formats=gztar,zip')
