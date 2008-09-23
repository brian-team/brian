'''
Runs all of the examples in the examples/ folder.

Note: removes all show() commands so that you don't have to wait for user input.
'''
import os, glob, sys, StringIO, time
import brian
os.chdir('../../examples')
examplefilenames = glob.glob('*.py')
examplefilenames = [ fname for fname in examplefilenames if 'parallelpythonised' not in fname ]
exceptions = []
starttime = time.time()
stdout = sys.stdout
stderr, sys.stderr = sys.stderr, StringIO.StringIO()
for fname in examplefilenames:
    try:
        code = open(fname,'r').read().replace('show()','')
        ns = {}
        print 'Running example', fname,
        brian.reinit_default_clock()
        sys.stdout = StringIO.StringIO()
        exec code in ns
        sys.stdout = stdout
        print '[ok]'
    except Exception, inst:
        sys.stdout = stdout
        print '[error]'
        exceptions.append((fname,inst))
sys.stderr = stderr
endtime = time.time()
print
if exceptions:
    print 'Exceptions encountered:'
    for fname, inst in exceptions:
        print 'In file', fname, '-', inst
    print
    print len(examplefilenames)-len(exceptions),'of',len(examplefilenames),'ran OK, time taken', endtime-starttime,'seconds'
else:
    print 'OK, time taken', endtime-starttime, 'seconds'