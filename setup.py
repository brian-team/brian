"""
Setup script for Brian
"""

from distutils.core import setup

from brian_setup_info import version

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
  package_data={'brian.utils.ccircular':['*.cxx', '*.h', '*.i', '*.cpp', '*.bat'],
                'brian.utils.fastexp':['*.cxx', '*.h', '*.i', '*.cpp', '*.bat', '*.c']},
  requires=['matplotlib(>=0.90.1)',
            'numpy(>=1.4.1)',
            'scipy(>=0.7.0)'
            ],
  url='http://www.briansimulator.org/',
  description='A clock-driven simulator for spiking neural networks',
  long_description=long_description,
  author='Romain Brette, Dan Goodman',
  author_email='Romain.Brette at ens.fr',
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
