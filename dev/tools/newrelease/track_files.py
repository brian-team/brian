'''
Track files that are or are not included in Brian releases.

The tracking data is stored in the folder tracking_data, and consists of
four files. The 'extras' files refer to the extras distribution and the
'release' files refer to the installation distribution. The 'rules' files give
the rules for which files should be tracked, and the 'files' files list the
files and whether or not they should be included or not. The rules files
consist of a series of regexps for which files should or shouldn't be included.
The regeps apply to the filename of the form 'brian/connection.py' (i.e. from
the brian root directory, using unix style forward slashes). The files.txt
files are a series of lines of the form 'C filename' where C is a character
which is ? if the file has not been tracked yet and requires user input, is
+ if the file should be included, - if the file shouldn't be included, and
* if the tracker should remind you whether or not it is included each time.
You can edit these files by hand.
'''
import os, sys, re, glob

pathname, filename = os.path.split(__file__)
os.chdir(pathname)
os.chdir('../../../')

#### List all the files in the Brian trunk

allfiles = []
for dirpath, dirs, files in os.walk('.'):
    if '.svn' in dirpath:
        continue
    dirs[:] = [_ for _ in dirs if '.svn' not in _]
    files = [dirpath + '/' + f for f in files]
    files = [f[2:].lower() for f in files]
    files = [f.replace('\\', '/') for f in files]
    allfiles.extend(files)

#### Extract the files that setup.py accounts for

setup = open('setup.py', 'r').read()
packages = eval(re.search(r'\bpackages\b\s*=\s*(\[.*?\])', setup, re.S).group(1).replace('\n', '').strip())
py_modules = eval(re.search(r'\bpy_modules\b\s*=\s*(\[.*?\])', setup, re.S).group(1).replace('\n', '').strip())
package_data = eval(re.search(r'\bpackage_data\b\s*=\s*(\{.*?\})', setup, re.S).group(1).replace('\n', '').strip())
extras_folders = eval(re.search(r'\bextras_folders\b\s*=\s*(\[.*?\])', setup, re.S).group(1).replace('\n', '').strip())

extras_files = []
for fold in extras_folders:
    files = glob.glob(fold)
    files = [f.replace('\\', '/').lower() for f in files]
    extras_files += files

release_files = ['readme.txt', 'setup.py']
for mod in py_modules:
    release_files.append(mod + '.py')
for p in packages:
    p = p.replace('.', '/')
    files = glob.glob(p + '/*.py')
    files = [f.replace('\\', '/').lower() for f in files]
    release_files.extend(files)
for p, pats in package_data.iteritems():
    p = p.replace('.', '/')
    for pat in pats:
        pat = p + '/' + pat
        files = glob.glob(pat)
        files = [f.replace('\\', '/').lower() for f in files]
        release_files.extend(files)

#### Load the data for given set of files

def apply_rules(rulesname, file_list):
    output_file_list = []
    for rule in open('dev/tools/newrelease/tracking_data/' + rulesname + '_rules.txt', 'r').read().split('\n'):
        if rule.startswith('include'):
            pattern = rule[8:]
            output_file_list.extend([f for f in file_list if re.search(pattern, f) and f not in output_file_list])
        elif rule.startswith('exclude'):
            pattern = rule[8:]
            output_file_list = [f for f in output_file_list if not re.search(pattern, f)]
        else:
            pass # assume it is a remark
    return output_file_list

def load_files(name, files):
    files = dict((f, '?') for f in files)
    for t in open('dev/tools/newrelease/tracking_data/' + name + '_files.txt', 'r').read().split('\n'):
        if t.strip():
            files[t[2:]] = t[0]
    # check unaccounted for files
    for f, c in files.items():
        if c == '?':
            print 'File', f, 'unaccounted for'
            print 'Choose action (+=should be included, -=should not, *=ask every time, enter to skip)'
            inp = raw_input('> ').lower().strip()
            if len(inp) == 0:
                continue
            if not inp == '+' and not inp == '*' and not inp == '-':
                print 'Bad input, skipping'
            files[f] = inp
    out = open('dev/tools/newrelease/tracking_data/' + name + '_files.txt', 'w')
    sorted_files = files.items()
    sorted_files.sort(key=lambda x:x[0])
    for f, c in sorted_files:
        print >> out, c, f
    out.close()
    return files

def get_files(name):
    files = apply_rules(name, allfiles)
    files = load_files(name, files)
    return files

#### Compare files

def compare_files(acc, files):
    ok = True
    strange = []
    too_many = []
    ask = []
    for f in acc:
        try:
            c = files[f]
            if c == '-':
                too_many.append(f)
            elif c == '*':
                ask.append(f)
            del files[f]
        except KeyError:
            strange.append(f)
    too_few = [f for f, c in files.iteritems() if c == '+']
    ask_not_inc = [f for f, c in files.iteritems() if c == '*']
    if len(strange):
        ok = False
        print 'Files included which are not tracked (check rules):'
        print
        for f in strange:
            print f
        print
    if len(too_many):
        ok = False
        print 'Files included which should not be:'
        print
        for f in too_many:
            print f
        print
    if len(ask):
        ok = False
        print 'Files included which you have marked to be asked about each time:'
        print
        for f in ask:
            print f
        print
    if len(too_few):
        ok = False
        print 'Files not included which should be:'
        print
        for f in too_few:
            print f
        print
    if len(ask_not_inc):
        ok = False
        print 'Files not included which you have marked to be asked about each time:'
        print
        for f in ask_not_inc:
            print f
        print
    return ok

print 'RELEASE'
print '======='
print

if compare_files(release_files, get_files('release')):
    print 'OK'
    print

print 'EXTRAS'
print '======'
print
if compare_files(extras_files, get_files('extras')):
    print 'OK'
    print

