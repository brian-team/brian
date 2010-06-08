'''
Inserts licence information in Brian modules.
'''

import os, re

os.chdir('../../../brian') # work from brian/ package directory

exclude = ['insert_licence.py', 'autodiff.py']

license = open('license.txt').read()
# Add comments
pattern = re.compile('^', flags=re.M)
license = pattern.sub('# ', license) + '\n'

def insert_license(filename):
    f_in = open(filename)
    text = f_in.read()
    f_in.close()

    if re.search('Copyright', text) is None:
        f_out = open(filename, 'w')
        f_out.write(license + text)
        f_out.close()
        return True
    else:
        return False

def insert_all(verbose=True):
    # Get all source files
    files = [file for file in os.listdir('') if file[-3:] == '.py' and (file not in exclude)] + \
          ['utils/' + file for file in os.listdir('utils') if file[-3:] == '.py' and (file not in exclude)]

    for file in files:
        if insert_license(file) and verbose:
            print file

if __name__ == '__main__':
    insert_all()
