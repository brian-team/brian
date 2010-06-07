'''
Runs all of the examples in the examples/ folder.

This version 

Note: removes all show() commands so that you don't have to wait for user input.
'''

import os, glob, sys, StringIO, time, gc, fnmatch
import brian

exclude_list=open('examples_exclude.txt', 'r').read().split('\n')
exclude_list=[f for f in exclude_list if not f.startswith('#') and f]

if os.path.exists('examples_completed.txt'):
    examples_completed=[f for f in open('examples_completed.txt', 'r').read().split('\n') if f]
else:
    examples_completed=[]

os.chdir('../../examples')


class GlobDirectoryWalker:
    # a forward iterator that traverses a directory tree

    def __init__(self, directory, pattern="*"):
        self.stack=[directory]
        self.pattern=pattern
        self.files=[]
        self.index=0

    def __getitem__(self, index):
        while 1:
            try:
                file=self.files[self.index]
                self.index=self.index+1
            except IndexError:
                # pop next directory from stack
                self.directory=self.stack.pop()
                self.files=os.listdir(self.directory)
                self.index=0
            else:
                # got a filename
                fullname=os.path.join(self.directory, file)
                if os.path.isdir(fullname) and not os.path.islink(fullname):
                    self.stack.append(fullname)
                if fnmatch.fnmatch(file, self.pattern):
                    return fullname

examplefilenames=[name for name in GlobDirectoryWalker('', '*.py') if all(fname not in name for fname in exclude_list)]
exceptions=[]
starttime=time.time()
stdout=sys.stdout
stderr, sys.stderr=sys.stderr, StringIO.StringIO()
for fname in examplefilenames:
    if fname not in examples_completed:
        try:
            code=open(fname, 'r').read().replace('show()', '')
            ns={}
            print 'Running example', fname,
            brian.reinit_default_clock()
            sys.stdout=StringIO.StringIO()
            exec code in ns
            del ns
            gc.collect()
            sys.stdout=stdout
            print '[ok]'
            examples_completed.append(fname)
            open('../dev/tests/examples_completed.txt', 'w').write('\n'.join(examples_completed))
        except Exception, inst:
            sys.stdout=stdout
            print '[error]'
            exceptions.append((fname, inst))
sys.stderr=stderr
endtime=time.time()
print
if exceptions:
    print 'Exceptions encountered:'
    for fname, inst in exceptions:
        print 'In file', fname, '-', inst
    print
    print len(examplefilenames)-len(exceptions), 'of', len(examplefilenames), 'ran OK, time taken', endtime-starttime, 'seconds'
else:
    print 'OK, time taken', endtime-starttime, 'seconds'
