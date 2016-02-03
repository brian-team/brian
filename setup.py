"""
Setup script for Brian
"""
import sys
import os
from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
from distutils.errors import CCompilerError

import numpy

# Insert the path of the directory where setup.py is located to sys.path,
# we need to import brian_setup_info from there
setup_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, setup_path)

from brian_setup_info import version

class optional_build_ext(build_ext):
    '''
    This class allows the building of C extensions to fail and still continue
    with the building process. This ensures that installation never fails, even
    on systems without a C compiler, for example.
    If brian is installed in an environment where building C extensions
    *should* work, set the environment variable BRIAN_SETUP_FAIL_ON_ERROR
    '''
    
    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except CCompilerError, ex:
            if os.getenv('BRIAN_SETUP_FAIL_ON_ERROR', False):
                raise ex
            else:
                error_msg = ('Building %s failed (see error message(s) '
                             'above) -- pure Python version will be used '
                             'instead.') % ext.name                             
                sys.stderr.write('*' * len(error_msg) + '\n' +
                                 error_msg + '\n' +
                                 '*' * len(error_msg) + '\n') 

long_description = '''
Brian is a simulator for spiking neural networks available on almost all platforms.
The motivation for this project is that a simulator should not only save the time of
processors, but also the time of scientists.

Brian is easy to learn and use, highly flexible and easily extensible. The Brian package
itself and simulations using it are all written in the Python programming language,
which is an easy, concise and highly developed language with many advanced features and
development tools, excellent documentation and a large community of users providing
support and extension packages.
'''



# Allow switching off extensions to allow building a pure Python Windows
# installer
ext_modules = []
if not os.getenv('BRIAN_SETUP_NO_EXTENSIONS', False):    
    utils_path = os.path.join('brian', 'utils')
    ext_modules.append(Extension('brian.utils.fastexp._fastexp',
                                 sources=[os.path.join(utils_path,
                                                       'fastexp', x) for x in
                                                       ('fastexp_wrap.cxx',
                                                        'fastexp.cpp',
                                                        'fexp.c')],
                                     include_dirs=[numpy.get_include()]
                                     ))
    ext_modules.append(Extension('brian.utils.ccircular._ccircular',
                                 sources=[os.path.join(utils_path,
                                                       'ccircular', x) for x in
                                                       ('ccircular_wrap.cxx',
                                                        'circular.cpp')],
                                     include_dirs=[numpy.get_include()]
                                     ))    


setup(name='brian',
  version=version,
  py_modules=['brian_unit_prefs', 'brian_no_units', 'brian_no_units_no_warnings'],
  packages=['brian',
                'brian.connections',
                'brian.synapses',
                'brian.deprecated',
                'brian.experimental',
                    'brian.experimental.cuda',
                    'brian.experimental.codegen',
                    'brian.experimental.codegen2',
                    'brian.experimental.codegen2.gpu',
                    'brian.experimental.compensation',
                    'brian.experimental.cspikequeue',
                    'brian.experimental.cuda',
                    'brian.experimental.genn',
                    'brian.experimental.nemo',
                    'brian.experimental.modelfitting',
                    'brian.experimental.neuromorphic',
                'brian.hears',
                    'brian.hears.filtering',
                    'brian.hears.hrtf',
                'brian.library',
                    'brian.library.electrophysiology',
                    'brian.library.modelfitting',
                'brian.tests',
                    'brian.tests.testcorrectness',
                    'brian.tests.testinterface',
                    'brian.tests.testutils',
                    'brian.tests.testfeatures',
                'brian.tools',
                'brian.utils',
                    'brian.utils.ccircular',
                    'brian.utils.fastexp',
                    'brian.utils.sparse_patch',
            ],
  ext_modules=ext_modules,
  # Add the source files for the CSpikeQueue as package data so they end up
  # being included in the installation directory for a possible later manual
  # build.
  package_data={'brian.experimental.cspikequeue': ['*.h', '*.c??']},
  cmdclass={'build_ext': optional_build_ext},
  provides=['brian'],
  requires=['matplotlib(>=0.90.1)',
            'numpy(>=1.4.1)',
            'scipy(>=0.7.0)'
            ],
  url='http://www.briansimulator.org/',
  description='A clock-driven simulator for spiking neural networks',
  long_description=long_description,
  author='Romain Brette, Dan Goodman',
  author_email='Romain.Brette at inserm.fr',
  download_url='https://neuralensemble.org/trac/brian/wiki/Downloads',
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'License :: Other/Proprietary License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering :: Bio-Informatics'
    ]
  )
