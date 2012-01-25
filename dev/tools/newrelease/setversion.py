'''
Tool to set the current version of Brian in the various places that have it,
i.e.:

* Global __version__ in __init__.py
* setup.py version
* docs version
* README.txt version
'''

import os, sys, brian, re
from update_release_date import setreleasedate

def setversion(version):
    docs_release = version
    major = docs_release[:docs_release.find('.')]
    minor = docs_release[docs_release.find('.') + 1:docs_release.find('.', docs_release.find('.') + 1)]
    docs_version = major + '.' + minor
    pathname = os.path.abspath(os.path.dirname(__file__))
    os.chdir(pathname)
    os.chdir('../../../')
    # update setup.py
    setup_info_py = open('brian_setup_info.py', 'r').read()
    setup_info_py = re.sub("version\s*=\s*'.*?'", "version = '" + version + "'", setup_info_py)
    open('brian_setup_info.py', 'w').write(setup_info_py)
    # update __init__.py
    init_py = open('brian/__init__.py', 'r').read()
    init_py = re.sub("__version__\s*=\s*'.*?'", "__version__ = '" + version + "'", init_py)
    open('brian/__init__.py', 'w').write(init_py)
    # update sphinx docs
    conf_py = open('docs_sphinx/conf.py', 'r').read()
    conf_py = re.sub("version\s*=\s*'.*?'", "version = '" + docs_version + "'", conf_py)
    conf_py = re.sub("release\s*=\s*'.*?'", "release = '" + docs_release + "'", conf_py)
    open('docs_sphinx/conf.py', 'w').write(conf_py)
    # update README.txt
    readme_txt = open('README.txt', 'r').read()
    readme_txt = re.sub(r"Version: .*\n", "Version: " + version + '\n', readme_txt)
    open('README.txt', 'w').write(readme_txt)

def user_input_setversion():
    print 'Current version of Brian is', brian.__version__
    version = raw_input('Enter new Brian version number: ')
    version = version.strip()
    print 'Changing to new version', version
    setversion(version)
    print 'Done'

if __name__ == '__main__':
    print 'Current version of Brian is', brian.__version__
    if len(sys.argv) < 2:
        version = raw_input('Enter new Brian version number: ')
    else:
        version = sys.argv[1]
    version = version.strip()
    print 'Changing to new version', version
    setversion(version)
    setreleasedate()
    print 'Done'
