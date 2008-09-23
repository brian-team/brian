'''
NOT FINISHED YET - DO NOT RUN!!!

Notes:

    Idea is to have all of the various configuration options for
    a release listed below, i.e. api excluded files, extras folders,
    modules, packages, version number, etc. This script then goes
    off and updates any files that need updating, and then builds a
    release based on that.

TODO:

* generate_api.py using the api_exclude
* update setup.py using the variables below
* update version number using the variable below (using setversion.py)
* stuff that is already in make_new_version() function
* have a new variable with files excluded from the release, and use
  this to alter the MANIFEST file
* build the distribution and extras file
* Add notes here about updating README.txt

Run this script to make a new release of Brian
'''

import os, sys, brian

version = '1.0.0rc2'

api_exclude = [ 'AEC', 'amplifier', 'circuits', 'electrodes',
           'autodiff', 'correlatedspikes' ]

extras_folders = ['tutorials/tutorial1_basic_concepts/*.py',
                  'tutorials/tutorial2_connections/*.py',
                  'examples/*.py',
                  'benchmarks/*.sce', 'benchmarks/*.m', 'benchmarks/*.cpp',
                  'docs/*.html', 'docs/images/*.jpg', 'docs/api/*.*' ]

py_modules= ['brian_unit_prefs','brian_no_units','brian_no_units_no_warnings']
packages = ['brian', 'brian.utils', 'brian.library', 'brian.tests']
url = 'http://brian.di.ens.fr/'
description = 'A clock-driven simulator for spiking neural networks'
author = 'Romain Brette, Dan Goodman'
author_email = 'Romain.Brette at ens.fr'

def make_new_release():
    basepathname, filename = os.path.split(__file__)
    os.chdir(basepathname)
    ####### STEP 1: MAKE SURE THE TESTS RUN ############
    tests_passed = brian.run_all_tests()
    if not tests_passed:
        contanyway = raw_input('\n\nTests failed, continue anyway (this is a bad idea)? [y/n] ')
        contanyway = contanyway.strip()
        if contanyway in ('n','N'):
            notcomplete()
            return
    ####### STEP 2: UPDATE THE VERSION NUMBER ###########
    os.chdir(basepathname)
    import setversion
    setversion.user_input_setversion()
    ####### STEP 3: NAGGING #############################
    ## NAG ABOUT API DOCUMENTATION
    print
    print 'At this point, you should make sure to edit the file'
    print '  dev/tools/docs/generate_api.py'
    print 'to accurately reflect the list of modules which should be'
    print 'excluded from the API documentation. When you have done'
    raw_input('this, press enter.')
    ## NAG ABOUT SETUP.PY
    print
    print 'At this point, you should update the setup.py file. If'
    print 'any new modules or packages have been introduced, they'
    print 'need to be added to the setup(...) function call. Also,'
    print 'any change in the list of files that should be included'
    print 'in the extras download should be made to this file.'
    print 'At this point, the documentation and the first stage of'
    print 'building the release will be done, this may take a minute'
    print 'or two. While it runs, you could update the README.txt'
    print 'file:'
    print '''In the base Brian directory is a file README.txt which should be
updated with the latest version numbers, etc. Also include here
a copy of the changes from brian/new_features.txt.'''
    raw_input('When you are ready to start the build, press enter.')
    ####### STEP 3: UPDATE THE DOCUMENTATION ############
    os.chdir(basepathname)
    os.chdir('../docs')
    import generate_html
    ####### STEP 4: UPDATE THE API DOCUMENTATION ########
    os.chdir(basepathname)
    os.chdir('../docs')
    import generate_api
    ####### STEP 5: CREATE DISTRIBUTION FILES INITIAL ###

def notcomplete():
    print '\n\n*** NEW RELEASE NOT COMPLETED ***'
    print
    raw_input('Press enter to finish.')

if __name__ == '__main__':
    make_new_release()