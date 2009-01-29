"""
Setup script for Brian
"""

from distutils.core import setup

version = '1.1.0'

long_description='''
Brian is a simulator for spiking neural networks available on almost all platforms.
The motivation for this project is that a simulator should not only save the time of
processors, but also the time of scientists.

Brian is easy to learn and use, highly flexible and easily extensible. The Brian package
itself and simulations using it are all written in the Python programming language,
which is an easy, concise and highly developed language with many advanced features and
development tools, excellent documentation and a large community of users providing
support and extension packages.
'''

# the create_extras.py script will automatically generate an extras files
# containing the following files
extras_folders = ['tutorials/tutorial1_basic_concepts/*.py',
                  'tutorials/tutorial2_connections/*.py',
                  'examples/*.py',
                  'benchmarks/*.sce', 'benchmarks/*.m', 'benchmarks/*.cpp',
                  'docs/*.*', 'docs/_images/*.jpg',# 'docs/api/*.*',
                  'docs/_sources/*.*', 'docs/_static/*.*' ]

if __name__=="__main__":
    setup(name='brian',
      version=version,
      py_modules=['brian_unit_prefs','brian_no_units','brian_no_units_no_warnings'],
      packages=['brian', 'brian.utils', 'brian.library', 'brian.tests', 'brian.experimental'],
      requires=['matplotlib(>=0.90.1)',
                'numpy(>=1.0.3)',
                'scipy(>=0.6.0)',
                'sympy(>=0.5.13)'
                ],
      url='http://brian.di.ens.fr/',
      description='A clock-driven simulator for spiking neural networks',
      long_description=long_description,
      author='Romain Brette, Dan Goodman',
      author_email='Romain.Brette at ens.fr',
      download_url='https://sourceforge.net/project/showfiles.php?group_id=226798',
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