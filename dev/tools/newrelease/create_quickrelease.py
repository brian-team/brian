import os, sys, datetime
from distutils.core import run_setup

import brian
from update_release_date import setreleasedate

pathname = os.path.abspath(os.path.dirname(__file__))
    
setreleasedate()

os.chdir(pathname)

os.chdir('../../../.') # work from Brian's root

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

run_setup('setup.py', ['bdist_wininst', '--plat-name=win32']) #to get the same file name on Linux and Windows
run_setup('setup.py', ['sdist', '--formats=gztar,zip'])
os.chdir('dist')
bname = 'brian-' + brian.__version__
bdate = str(datetime.date.today())
    
for ext in ['tar.gz', 'zip', 'win32.exe']:
    fname = bname + '-' + bdate + '.' + ext
    if os.path.exists(fname): 
        print 'Deleting "%s"' % fname
        os.remove(fname)
    
    oldfname = '%s.%s' % (bname, ext)
    print 'Renaming "%s" to "%s"' % (oldfname, fname)
    os.rename(oldfname, fname)    
