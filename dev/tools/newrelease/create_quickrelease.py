import os, brian, datetime
from update_release_date import setreleasedate
setreleasedate()
pathname, filename = os.path.split(__file__)
os.chdir(pathname)
os.chdir('../../../.') # work from Brian's root
os.system('del MANIFEST')
os.system('setup.py bdist_wininst')
os.system('setup.py sdist --formats=gztar,zip')
os.chdir('dist')
bname = 'brian-' + brian.__version__
bdate = str(datetime.date.today())
def exec_command(cmd):
    print 'Executing command:', cmd
    os.system(cmd)
for ext in ['tar.gz', 'zip', 'win32.exe']:
    if os.path.exists(bname + '-' + bdate + '.' + ext):
        exec_command('del %s-%s.%s' % (bname, bdate, ext))
    exec_command('rename %s.%s %s-%s.%s' % (bname, ext, bname, bdate, ext))
