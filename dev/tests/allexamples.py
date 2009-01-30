'''
Runs all of the examples in the examples/ folder.

Note: removes all show() commands so that you don't have to wait for user input.
'''
import os, glob, sys, StringIO, time, gc, fnmatch
import brian
os.chdir('../../examples')

class GlobDirectoryWalker:
    # a forward iterator that traverses a directory tree

    def __init__(self, directory, pattern="*"):
        self.stack = [directory]
        self.pattern = pattern
        self.files = []
        self.index = 0

    def __getitem__(self, index):
        while 1:
            try:
                file = self.files[self.index]
                self.index = self.index + 1
            except IndexError:
                # pop next directory from stack
                self.directory = self.stack.pop()
                self.files = os.listdir(self.directory)
                self.index = 0
            else:
                # got a filename
                fullname = os.path.join(self.directory, file)
                if os.path.isdir(fullname) and not os.path.islink(fullname):
                    self.stack.append(fullname)
                if fnmatch.fnmatch(file, self.pattern):
                    return fullname

exclude_list=['parallelpython.py','pickle_loadnet.py','pickle_savenet.py','stim2d.py',
              'interface.py','STDP1','STDP2']
examplefilenames = [name for name in GlobDirectoryWalker('','*.py') if all(fname not in name for fname in exclude_list)]
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
        del ns
        gc.collect()
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